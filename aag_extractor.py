"""
AAG (Attributed Adjacency Graph) Extractor
从STEP文件中提取几何属性邻接图
"""
import argparse
from multiprocessing.pool import Pool
import gc
import json
import os.path as osp
import numpy as np
from pathlib import Path
from tqdm import tqdm
from itertools import repeat

from OCC.Core.BRep import BRep_Tool
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Extend import TopologyUtils
from OCC.Core.TopAbs import TopAbs_IN, TopAbs_FORWARD, TopAbs_REVERSED
from OCC.Core.TopAbs import (TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE, TopAbs_WIRE,
                             TopAbs_SHELL, TopAbs_SOLID, TopAbs_COMPOUND,
                             TopAbs_COMPSOLID)
from OCC.Core.TopoDS import (
    TopoDS_Solid,
    TopoDS_Compound,
    TopoDS_CompSolid,
)
from OCC.Core.TopExp import topexp
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop_LinearProperties, brepgprop_SurfaceProperties
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCC.Core.GeomAbs import (GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
                              GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BezierSurface,
                              GeomAbs_BSplineSurface, GeomAbs_Line, GeomAbs_Circle,
                              GeomAbs_Ellipse, GeomAbs_Hyperbola, GeomAbs_Parabola,
                              GeomAbs_BezierCurve, GeomAbs_BSplineCurve,
                              GeomAbs_OffsetCurve, GeomAbs_OtherCurve)

# occwl
from occwl.edge_data_extractor import EdgeDataExtractor, EdgeConvexity
from occwl.edge import Edge
from occwl.face import Face
from occwl.solid import Solid
from occwl.uvgrid import uvgrid
from occwl.graph import face_adjacency


def scale_solid_to_unit_box(solid):
    if isinstance(solid, Solid):
        return solid.scale_to_unit_box(copy=True)
    solid = Solid(solid, allow_compound=True)
    solid = solid.scale_to_unit_box(copy=True)
    return solid.topods_shape()


class TopologyChecker:
    """
    拓扑检查器，验证STEP文件的几何拓扑是否有效
    """
    def __init__(self):
        pass

    def find_edges_from_wires(self, top_exp):
        edge_set = set()
        for wire in top_exp.wires():
            wire_exp = TopologyUtils.WireExplorer(wire)
            for edge in wire_exp.ordered_edges():
                edge_set.add(edge)
        return edge_set

    def find_edges_from_top_exp(self, top_exp):
        edge_set = set(top_exp.edges())
        return edge_set

    def check_closed(self, body):
        top_exp = TopologyUtils.TopologyExplorer(body, ignore_orientation=False)
        edges_from_wires = self.find_edges_from_wires(top_exp)
        edges_from_top_exp = self.find_edges_from_top_exp(top_exp)
        missing_edges = edges_from_top_exp - edges_from_wires
        return len(missing_edges) == 0

    def check_manifold(self, top_exp):
        faces = set()
        for shell in top_exp.shells():
            for face in top_exp._loop_topo(TopAbs_FACE, shell):
                if face in faces:
                    return False
                faces.add(face)
        return True

    def check_unique_coedges(self, top_exp):
        coedge_set = set()
        for loop in top_exp.wires():
            wire_exp = TopologyUtils.WireExplorer(loop)
            for coedge in wire_exp.ordered_edges():
                orientation = coedge.Orientation()
                tup = (coedge, orientation)
                if tup in coedge_set:
                    return False
                coedge_set.add(tup)
        return True

    def __call__(self, body):
        top_exp = TopologyUtils.TopologyExplorer(body, ignore_orientation=True)
        if top_exp.number_of_faces() == 0:
            print('Empty shape')
            return False
        analyzer = BRepCheck_Analyzer(body)
        if not analyzer.IsValid(body):
            print('BRepCheck_Analyzer found defects')
            return False
        if not self.check_manifold(top_exp):
            print("Non-manifold bodies are not supported")
            return False
        if not self.check_closed(body):
            print("Bodies which are not closed are not supported")
            return False
        if not self.check_unique_coedges(top_exp):
            print("Bodies where the same coedge is uses in multiple loops are not supported")
            return False
        return True


class AAGExtractor:
    """
    从STEP文件提取几何属性邻接图(AAG)的主类
    """
    def __init__(
        self,
        step_file,
        attribute_schema=None,
        scale_body=True
    ):
        """
        Args:
            step_file: STEP文件路径
            attribute_schema: 属性定义字典，如果为None则使用默认schema
            scale_body: 是否缩放实体到单位立方体
        """
        self.step_file = step_file
        self.scale_body = scale_body

        # 默认属性schema
        if attribute_schema is None:
            self.attribute_schema = self._default_schema()
        else:
            self.attribute_schema = attribute_schema

        # 是否使用UV grid
        self.use_uv = "UV-grid" in self.attribute_schema.keys()
        self.topchecker = TopologyChecker()

        if self.use_uv:
            self.num_srf_u = self.attribute_schema["UV-grid"]["num_srf_u"]
            self.num_srf_v = self.attribute_schema["UV-grid"]["num_srf_v"]
            self.num_crv_u = self.attribute_schema["UV-grid"]["num_crv_u"]

    @staticmethod
    def _default_schema():
        """默认的属性schema"""
        return {
            "face_attributes": [
                "Plane",
                "Cylinder",
                "Cone",
                "SphereFaceAttribute",
                "TorusFaceAttribute",
                "FaceAreaAttribute",
                "RationalNurbsFaceAttribute",
                "FaceCentroidAttribute"
            ],
            "edge_attributes": [
                "Concave edge",
                "Convex edge",
                "Smooth",
                "EdgeLengthAttribute",
                "CircularEdgeAttribute",
                "ClosedEdgeAttribute",
                "EllipticalEdgeAttribute",
                "NonRationalBSplineEdgeAttribute",
                "RationalBSplineEdgeAttribute",
                "StraightEdgeAttribute"
            ],
            "UV-grid": {
                "num_srf_u": 5,
                "num_srf_v": 5,
                "num_crv_u": 0
            }
        }

    def process(self):
        """
        执行AAG提取

        Returns:
            dict: 包含图结构和属性的字典
        """
        self.body = self.load_body_from_step()
        assert self.body is not None, \
            f"the shape {self.step_file} is non-manifold or open"
        assert self.topchecker(self.body), \
            f"the shape {self.step_file} has wrong topology"
        assert isinstance(self.body, TopoDS_Solid), \
            f'file {self.step_file} is {type(self.body)}, not TopoDS_Solid'

        if self.scale_body:
            self.body = scale_solid_to_unit_box(self.body)

        try:
            graph = face_adjacency(Solid(self.body))
        except Exception as e:
            print(e)
            assert False, f'Wrong shape {self.step_file}'

        graph_face_attr = []
        graph_face_grid = []
        len_of_face_attr = len(self.attribute_schema["face_attributes"]) + \
            (2 if "FaceCentroidAttribute" in self.attribute_schema["face_attributes"] else 0)

        for face_idx in graph.nodes:
            face = graph.nodes[face_idx]["face"]
            face_attr = self.extract_attributes_from_face(face.topods_shape())
            assert len_of_face_attr == len(face_attr)
            graph_face_attr.append(face_attr)

            if self.use_uv and self.num_srf_u and self.num_srf_v:
                uv_grid = self.extract_face_point_grid(face)
                assert uv_grid.shape[0] == 7
                graph_face_grid.append(uv_grid.tolist())

        graph_edge_attr = []
        graph_edge_grid = []
        for edge_idx in graph.edges:
            edge = graph.edges[edge_idx]["edge"]
            if not edge.has_curve():
                continue
            edge = edge.topods_shape()
            edge_attr = self.extract_attributes_from_edge(edge)
            assert len(self.attribute_schema["edge_attributes"]) == len(edge_attr)
            graph_edge_attr.append(edge_attr)

            if self.use_uv and self.num_crv_u:
                u_grid = self.extract_edge_point_grid(edge)
                assert u_grid.shape[0] == 12
                graph_edge_grid.append(u_grid.tolist())

        edges = list(graph.edges)
        src = [e[0] for e in edges]
        dst = [e[1] for e in edges]
        graph = {
            'edges': (src, dst),
            'num_nodes': len(graph.nodes)
        }

        return {
            'graph': graph,
            'graph_face_attr': graph_face_attr,
            'graph_face_grid': graph_face_grid,
            'graph_edge_attr': graph_edge_attr,
            'graph_edge_grid': graph_edge_grid,
        }

    def load_body_from_step(self):
        step_filename_str = str(self.step_file)
        reader = STEPControl_Reader()
        reader.ReadFile(step_filename_str)
        reader.TransferRoots()
        shape = reader.OneShape()
        return shape

    def extract_attributes_from_face(self, face):
        def plane_attribute(face):
            surf_type = BRepAdaptor_Surface(face).GetType()
            return 1.0 if surf_type == GeomAbs_Plane else 0.0

        def cylinder_attribute(face):
            surf_type = BRepAdaptor_Surface(face).GetType()
            return 1.0 if surf_type == GeomAbs_Cylinder else 0.0

        def cone_attribute(face):
            surf_type = BRepAdaptor_Surface(face).GetType()
            return 1.0 if surf_type == GeomAbs_Cone else 0.0

        def sphere_attribute(face):
            surf_type = BRepAdaptor_Surface(face).GetType()
            return 1.0 if surf_type == GeomAbs_Sphere else 0.0

        def torus_attribute(face):
            surf_type = BRepAdaptor_Surface(face).GetType()
            return 1.0 if surf_type == GeomAbs_Torus else 0.0

        def area_attribute(face):
            geometry_properties = GProp_GProps()
            brepgprop_SurfaceProperties(face, geometry_properties)
            return geometry_properties.Mass()

        def rational_nurbs_attribute(face):
            surf = BRepAdaptor_Surface(face)
            if surf.GetType() == GeomAbs_BSplineSurface:
                bspline = surf.BSpline()
            elif surf.GetType() == GeomAbs_BezierSurface:
                bspline = surf.Bezier()
            else:
                bspline = None

            if bspline is not None:
                if bspline.IsURational() or bspline.IsVRational():
                    return 1.0
            return 0.0

        def centroid_attribute(face):
            mass_props = GProp_GProps()
            brepgprop_SurfaceProperties(face, mass_props)
            gPt = mass_props.CentreOfMass()
            return gPt.Coord()

        face_attributes = []
        for attribute in self.attribute_schema["face_attributes"]:
            if attribute == "Plane":
                face_attributes.append(plane_attribute(face))
            elif attribute == "Cylinder":
                face_attributes.append(cylinder_attribute(face))
            elif attribute == "Cone":
                face_attributes.append(cone_attribute(face))
            elif attribute == "SphereFaceAttribute":
                face_attributes.append(sphere_attribute(face))
            elif attribute == "TorusFaceAttribute":
                face_attributes.append(torus_attribute(face))
            elif attribute == "FaceAreaAttribute":
                face_attributes.append(area_attribute(face))
            elif attribute == "RationalNurbsFaceAttribute":
                face_attributes.append(rational_nurbs_attribute(face))
            elif attribute == "FaceCentroidAttribute":
                face_attributes.extend(centroid_attribute(face))
            else:
                assert False, f"Unknown face attribute: {attribute}"
        return face_attributes

    def extract_attributes_from_edge(self, edge):
        def find_edge_convexity(edge, faces):
            edge_data = EdgeDataExtractor(Edge(edge),
                faces, use_arclength_params=False)
            if not edge_data.good:
                return 0.0
            angle_tol_rads = 0.0872664626
            convexity = edge_data.edge_convexity(angle_tol_rads)
            return convexity

        def convexity_attribute(convexity, attribute):
            if attribute == "Convex edge":
                return convexity == EdgeConvexity.CONVEX
            if attribute == "Concave edge":
                return convexity == EdgeConvexity.CONCAVE
            if attribute == "Smooth":
                return convexity == EdgeConvexity.SMOOTH
            assert False, f"Unknown convexity: {attribute}"
            return 0.0

        def edge_length_attribute(edge):
            geometry_properties = GProp_GProps()
            brepgprop_LinearProperties(edge, geometry_properties)
            return geometry_properties.Mass()

        def circular_edge_attribute(edge):
            brep_adaptor_curve = BRepAdaptor_Curve(edge)
            curv_type = brep_adaptor_curve.GetType()
            return 1.0 if curv_type == GeomAbs_Circle else 0.0

        def closed_edge_attribute(edge):
            return 1.0 if BRep_Tool().IsClosed(edge) else 0.0

        def elliptical_edge_attribute(edge):
            brep_adaptor_curve = BRepAdaptor_Curve(edge)
            curv_type = brep_adaptor_curve.GetType()
            return 1.0 if curv_type == GeomAbs_Ellipse else 0.0

        def straight_edge_attribute(edge):
            brep_adaptor_curve = BRepAdaptor_Curve(edge)
            curv_type = brep_adaptor_curve.GetType()
            return 1.0 if curv_type == GeomAbs_Line else 0.0

        def bezier_edge_attribute(edge):
            return 1.0 if Edge(edge).curve_type() == "bezier" else 0.0

        def non_rational_bspline_edge_attribute(edge):
            occwl_edge = Edge(edge)
            return 1.0 if (occwl_edge.curve_type() == "bspline" and not occwl_edge.rational()) else 0.0

        def rational_bspline_edge_attribute(edge):
            occwl_edge = Edge(edge)
            return 1.0 if (occwl_edge.curve_type() == "bspline" and occwl_edge.rational()) else 0.0

        top_exp = TopologyUtils.TopologyExplorer(self.body, ignore_orientation=True)
        faces_of_edge = [Face(f) for f in top_exp.faces_from_edge(edge)]

        attribute_list = self.attribute_schema["edge_attributes"]
        if "Concave edge" in attribute_list or \
            "Convex edge" in attribute_list or \
            "Smooth" in attribute_list:
            convexity = find_edge_convexity(edge, faces_of_edge)

        edge_attributes = []
        for attribute in attribute_list:
            if attribute == "Concave edge":
                edge_attributes.append(convexity_attribute(convexity, attribute))
            elif attribute == "Convex edge":
                edge_attributes.append(convexity_attribute(convexity, attribute))
            elif attribute == "Smooth":
                edge_attributes.append(convexity_attribute(convexity, attribute))
            elif attribute == "EdgeLengthAttribute":
                edge_attributes.append(edge_length_attribute(edge))
            elif attribute == "CircularEdgeAttribute":
                edge_attributes.append(circular_edge_attribute(edge))
            elif attribute == "ClosedEdgeAttribute":
                edge_attributes.append(closed_edge_attribute(edge))
            elif attribute == "EllipticalEdgeAttribute":
                edge_attributes.append(elliptical_edge_attribute(edge))
            elif attribute == "StraightEdgeAttribute":
                edge_attributes.append(straight_edge_attribute(edge))
            elif attribute == "BezierEdgeAttribute":
                edge_attributes.append(bezier_edge_attribute(edge))
            elif attribute == "NonRationalBSplineEdgeAttribute":
                edge_attributes.append(non_rational_bspline_edge_attribute(edge))
            elif attribute == "RationalBSplineEdgeAttribute":
                edge_attributes.append(rational_bspline_edge_attribute(edge))
            else:
                assert False, f"Unknown edge attribute: {attribute}"
        return edge_attributes

    def extract_face_point_grid(self, face):
        points = uvgrid(face, self.num_srf_u, self.num_srf_v, method="point")
        normals = uvgrid(face, self.num_srf_u, self.num_srf_v, method="normal")
        mask = uvgrid(face, self.num_srf_u, self.num_srf_v, method="inside")

        single_grid = np.concatenate([points, normals, mask], axis=2)
        return np.transpose(single_grid, (2, 0, 1))

    def extract_edge_point_grid(self, edge):
        top_exp = TopologyUtils.TopologyExplorer(self.body, ignore_orientation=True)
        faces_of_edge = [Face(f) for f in top_exp.faces_from_edge(edge)]

        edge_data = EdgeDataExtractor(Edge(edge), faces_of_edge,
            num_samples=self.num_crv_u, use_arclength_params=True)
        if not edge_data.good:
            return np.zeros((12, self.num_crv_u))

        single_grid = np.concatenate(
            [
                edge_data.points,
                edge_data.tangents,
                edge_data.left_normals,
                edge_data.right_normals
            ],
            axis = 1
        )
        return np.transpose(single_grid, (1, 0))


def find_standardization(data):
    """
    计算属性的均值和标准差用于归一化
    """
    all_face_attr = []
    all_edge_attr = []
    for one_sample in data:
        fn, graph = one_sample
        all_face_attr.extend(graph["graph_face_attr"])
        all_edge_attr.extend(graph["graph_edge_attr"])
    graph_face_attr = np.asarray(all_face_attr)
    graph_edge_attr = np.asarray(all_edge_attr)

    mean_face_attr = np.mean(graph_face_attr, axis=0)
    std_face_attr = np.std(graph_face_attr, axis=0)

    mean_edge_attr = np.mean(graph_edge_attr, axis=0)
    std_edge_attr = np.std(graph_edge_attr, axis=0)

    return {
        'mean_face_attr': mean_face_attr.tolist(),
        'std_face_attr': std_face_attr.tolist(),
        'mean_edge_attr': mean_edge_attr.tolist(),
        'std_edge_attr': std_edge_attr.tolist(),
    }


def save_json_data(pathname, data):
    with open(pathname, 'w', encoding='utf8') as fp:
        json.dump(data, fp, indent=4, ensure_ascii=False, sort_keys=False)


def load_json(pathname):
    with open(pathname, "r") as fp:
        return json.load(fp)


def initializer():
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def process_one_file(args):
    fn, attribute_schema = args
    extractor = AAGExtractor(fn, attribute_schema)
    out = extractor.process()
    return [str(fn.stem), out]


def extract_aag_from_step(step_path, output_path, attribute_schema=None, num_workers=1):
    """
    批量从STEP文件提取AAG

    Args:
        step_path: STEP文件目录
        output_path: 输出目录
        attribute_schema: 可选的属性schema
        num_workers: 并行进程数
    """
    step_path = Path(step_path)
    output_path = Path(output_path)
    if not output_path.exists():
        output_path.mkdir(parents=True)

    if attribute_schema is None:
        attribute_schema = AAGExtractor._default_schema()

    step_files = list(step_path.glob("*.st*p"))

    if num_workers == 1:
        results = []
        for fn in tqdm(step_files):
            try:
                result = process_one_file((fn, attribute_schema))
                results.append(result)
            except Exception as e:
                print(f"Failed to process {fn}: {e}")
    else:
        pool = Pool(processes=num_workers, initializer=initializer)
        try:
            results = list(tqdm(
                pool.imap(
                    process_one_file, zip(step_files, repeat(attribute_schema))),
                total=len(step_files)))
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()

    save_json_data(osp.join(output_path, 'graphs.json'), results)

    if results:
        attr_stat = find_standardization(results)
        save_json_data(osp.join(output_path, 'attr_stat.json'), attr_stat)

    gc.collect()
    print(f"Processed {len(results)} files.")


def main():
    parser = argparse.ArgumentParser(description='Extract AAG from STEP files')
    parser.add_argument("--step_path", type=str, required=True, help="Path to STEP files")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--schema", type=str, required=False, help="Optional attribute schema JSON")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers")
    args = parser.parse_args()

    attribute_schema = None
    if args.schema:
        attribute_schema = load_json(args.schema)

    extract_aag_from_step(args.step_path, args.output, attribute_schema, args.num_workers)


if __name__ == '__main__':
    main()
