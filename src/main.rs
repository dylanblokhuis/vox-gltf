use std::{fs, io::Write};

use block_mesh::{
    greedy_quads,
    ndshape::{ConstShape, ConstShape3u32},
    Axis, AxisPermutation, GreedyQuadsBuffer, MergeVoxel, OrientedBlockFace, QuadCoordinateConfig,
    Voxel, VoxelVisibility,
};
use dot_vox::load;

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct BoolVoxel(pub u8);

pub const EMPTY_VOXEL: BoolVoxel = BoolVoxel(255);

impl Voxel for BoolVoxel {
    fn get_visibility(&self) -> VoxelVisibility {
        if self.0 == 255 {
            VoxelVisibility::Empty
        } else {
            VoxelVisibility::Opaque
        }
    }
}

impl MergeVoxel for BoolVoxel {
    type MergeValue = Self;
    fn merge_value(&self) -> Self::MergeValue {
        *self
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum Output {
    /// Output standard glTF.
    Standard,

    /// Output binary glTF.
    Binary,
}

#[repr(C, align(4))]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    color: [u8; 4],
}

/// Calculate bounding coordinates of a list of vertices, used for the clipping distance of the model
fn bounding_coords(points: &[Vertex]) -> ([f32; 3], [f32; 3]) {
    let mut min = [f32::MAX, f32::MAX, f32::MAX];
    let mut max = [f32::MIN, f32::MIN, f32::MIN];

    for point in points {
        let p = point.position;
        for i in 0..3 {
            min[i] = f32::min(min[i], p[i]);
            max[i] = f32::max(max[i], p[i]);
        }
    }
    (min, max)
}

/// Coordinate configuration for a right-handed coordinate system with Z up.
///
/// ```text
///       +Z      
///       | -Y    
/// -X____|/____+X
///      /|       
///    +Y |       
///       -Z      
/// ```
pub const RIGHT_HANDED_Z_UP_CONFIG: QuadCoordinateConfig = QuadCoordinateConfig {
    // Z is always in the V direction when it's not the normal. When Z is the
    // normal, right-handedness determines that we must use Zxy permutations.
    faces: [
        OrientedBlockFace::new(-1, AxisPermutation::Xyz),
        OrientedBlockFace::new(-1, AxisPermutation::Zxy),
        OrientedBlockFace::new(-1, AxisPermutation::Yzx),
        OrientedBlockFace::new(1, AxisPermutation::Xyz),
        OrientedBlockFace::new(1, AxisPermutation::Zxy),
        OrientedBlockFace::new(1, AxisPermutation::Yzx),
    ],
    u_flip_face: Axis::X,
};
const MAX_SIZE: u32 = 258;
type ChunkShape = ConstShape3u32<MAX_SIZE, MAX_SIZE, MAX_SIZE>;
fn main() {
    let vox = load("./room.vox").unwrap();
    let model = &vox.models[0];

    println!("{:?}", model.size);
    let mut voxels = vec![EMPTY_VOXEL; ChunkShape::SIZE as usize];
    println!("{:?} {}", voxels.len(), model.voxels.len());

    for voxel in &model.voxels {
        let x = voxel.x as u32 + 1;
        let y = voxel.y as u32 + 1;
        let z = voxel.z as u32 + 1;

        voxels[(x + z * MAX_SIZE + y * MAX_SIZE * MAX_SIZE) as usize] = BoolVoxel(voxel.i)
    }

    let mut buffer = GreedyQuadsBuffer::new(voxels.len());

    let faces = RIGHT_HANDED_Z_UP_CONFIG.faces;
    greedy_quads(
        &voxels,
        &ChunkShape {},
        [0; 3],
        [MAX_SIZE - 1; 3],
        &faces,
        &mut buffer,
    );

    let num_indices = buffer.quads.num_quads() * 6;
    let num_vertices = buffer.quads.num_quads() * 4;

    let mut indices = Vec::with_capacity(num_indices);
    let mut positions = Vec::with_capacity(num_vertices);
    let mut normals = Vec::with_capacity(num_vertices);
    let mut vertices = Vec::with_capacity(num_vertices);
    let mut colors = Vec::with_capacity(num_vertices);

    for (group, face) in buffer.quads.groups.into_iter().zip(faces.into_iter()) {
        for quad in group.into_iter() {
            let palette_index = voxels[ChunkShape::linearize(quad.minimum) as usize].0;
            colors.extend_from_slice(&[vox.palette[palette_index as usize]; 4]);
            indices.extend_from_slice(&face.quad_mesh_indices(positions.len() as u32));
            positions.extend_from_slice(&face.quad_mesh_positions(&quad, 1.0));
            normals.extend_from_slice(&face.quad_mesh_normals());
        }
    }

    println!("{:?} {:?}", positions.len(), indices.len());

    for i in 0..num_vertices {
        vertices.push(Vertex {
            position: [
                positions[i][0] as f32,
                positions[i][1] as f32,
                positions[i][2] as f32,
            ],
            normal: [
                normals[i][0] as f32,
                normals[i][1] as f32,
                normals[i][2] as f32,
            ],
            color: [
                colors[i].r as u8,
                colors[i].g as u8,
                colors[i].b as u8,
                colors[i].a as u8,
            ],
        });
    }

    let (min, max) = bounding_coords(&vertices);
    let vertex_buffer_length = (vertices.len() * std::mem::size_of::<Vertex>()) as u32;
    let indices_buffer_length = (indices.len() * std::mem::size_of::<u32>()) as u32;
    let output = Output::Standard;

    let vertex_buffer = gltf_json::Buffer {
        byte_length: vertex_buffer_length,
        extensions: Default::default(),
        extras: Default::default(),
        name: None,
        uri: if output == Output::Standard {
            Some("buffer0.bin".into())
        } else {
            None
        },
    };

    let indices_buffer = gltf_json::Buffer {
        byte_length: indices_buffer_length,
        extensions: Default::default(),
        extras: Default::default(),
        name: None,
        uri: if output == Output::Standard {
            Some("buffer1.bin".into())
        } else {
            None
        },
    };

    let vertex_buffer_view = gltf_json::buffer::View {
        buffer: gltf_json::Index::new(0),
        byte_length: vertex_buffer.byte_length,
        byte_offset: None,
        byte_stride: Some(std::mem::size_of::<Vertex>() as u32),
        extensions: Default::default(),
        extras: Default::default(),
        name: None,
        target: Some(gltf_json::validation::Checked::Valid(
            gltf_json::buffer::Target::ArrayBuffer,
        )),
    };

    let indices_buffer_view = gltf_json::buffer::View {
        buffer: gltf_json::Index::new(1),
        byte_length: indices_buffer.byte_length,
        byte_offset: None,
        byte_stride: Some(std::mem::size_of::<u32>() as u32),
        extensions: Default::default(),
        extras: Default::default(),
        name: None,
        target: Some(gltf_json::validation::Checked::Valid(
            gltf_json::buffer::Target::ElementArrayBuffer,
        )),
    };

    let positions = gltf_json::Accessor {
        buffer_view: Some(gltf_json::Index::new(0)),
        byte_offset: Some(0),
        count: vertices.len() as u32,
        component_type: gltf_json::validation::Checked::Valid(
            gltf_json::accessor::GenericComponentType(gltf_json::accessor::ComponentType::F32),
        ),
        extensions: Default::default(),
        extras: Default::default(),
        type_: gltf_json::validation::Checked::Valid(gltf_json::accessor::Type::Vec3),
        min: Some(gltf_json::Value::from(Vec::from(min))),
        max: Some(gltf_json::Value::from(Vec::from(max))),
        name: None,
        normalized: false,
        sparse: None,
    };
    let normal = gltf_json::Accessor {
        buffer_view: Some(gltf_json::Index::new(0)),
        byte_offset: Some((3 * std::mem::size_of::<f32>()) as u32),
        count: vertices.len() as u32,
        component_type: gltf_json::validation::Checked::Valid(
            gltf_json::accessor::GenericComponentType(gltf_json::accessor::ComponentType::F32),
        ),
        extensions: Default::default(),
        extras: Default::default(),
        type_: gltf_json::validation::Checked::Valid(gltf_json::accessor::Type::Vec3),
        min: None,
        max: None,
        name: None,
        normalized: false,
        sparse: None,
    };

    let colors = gltf_json::Accessor {
        buffer_view: Some(gltf_json::Index::new(0)),
        byte_offset: Some((6 * std::mem::size_of::<f32>()) as u32),
        count: vertices.len() as u32,
        component_type: gltf_json::validation::Checked::Valid(
            gltf_json::accessor::GenericComponentType(gltf_json::accessor::ComponentType::U8),
        ),
        extensions: Default::default(),
        extras: Default::default(),
        type_: gltf_json::validation::Checked::Valid(gltf_json::accessor::Type::Vec4),
        min: None,
        max: None,
        name: None,
        normalized: true,
        sparse: None,
    };

    let indices_accessor = gltf_json::Accessor {
        buffer_view: Some(gltf_json::Index::new(1)),
        byte_offset: Some(0),
        count: indices.len() as u32,
        component_type: gltf_json::validation::Checked::Valid(
            gltf_json::accessor::GenericComponentType(gltf_json::accessor::ComponentType::U32),
        ),
        extensions: Default::default(),
        extras: Default::default(),
        type_: gltf_json::validation::Checked::Valid(gltf_json::accessor::Type::Scalar),
        min: None,
        max: None,
        name: None,
        normalized: false,
        sparse: None,
    };

    let primitive = gltf_json::mesh::Primitive {
        attributes: {
            let mut map = std::collections::BTreeMap::new();
            map.insert(
                gltf_json::validation::Checked::Valid(gltf_json::mesh::Semantic::Positions),
                gltf_json::Index::new(0),
            );
            map.insert(
                gltf_json::validation::Checked::Valid(gltf_json::mesh::Semantic::Normals),
                gltf_json::Index::new(1),
            );
            map.insert(
                gltf_json::validation::Checked::Valid(gltf_json::mesh::Semantic::Colors(0)),
                gltf_json::Index::new(2),
            );

            map
        },
        extensions: Default::default(),
        extras: Default::default(),
        indices: Some(gltf_json::Index::new(3)),
        material: None,
        mode: gltf_json::validation::Checked::Valid(gltf_json::mesh::Mode::Triangles),
        targets: None,
    };

    let mesh = gltf_json::Mesh {
        extensions: Default::default(),
        extras: Default::default(),
        name: None,
        primitives: vec![primitive],
        weights: None,
    };

    let node = gltf_json::Node {
        camera: None,
        children: None,
        extensions: Default::default(),
        extras: Default::default(),
        matrix: None,
        mesh: Some(gltf_json::Index::new(0)),
        name: None,
        rotation: None,
        scale: None,
        translation: None,
        skin: None,
        weights: None,
    };

    let root = gltf_json::Root {
        accessors: vec![positions, normal, colors, indices_accessor],
        buffers: vec![vertex_buffer, indices_buffer],
        buffer_views: vec![vertex_buffer_view, indices_buffer_view],
        meshes: vec![mesh],
        nodes: vec![node],
        scenes: vec![gltf_json::Scene {
            extensions: Default::default(),
            extras: Default::default(),
            name: None,
            nodes: vec![gltf_json::Index::new(0)],
        }],
        ..Default::default()
    };

    match output {
        Output::Standard => {
            let _ = fs::create_dir("output");

            let writer = fs::File::create("output/output.gltf").expect("I/O error");
            gltf_json::serialize::to_writer_pretty(writer, &root).expect("Serialization error");

            let mut writer = fs::File::create("output/buffer0.bin").expect("I/O error");
            writer
                .write_all(bytemuck::cast_slice(&vertices))
                .expect("I/O error");

            let mut writer = fs::File::create("output/buffer1.bin").expect("I/O error");
            writer
                .write_all(bytemuck::cast_slice(&indices))
                .expect("I/O error");
        }
        Output::Binary => {
            // let json_string = gltf_json::serialize::to_string(&root).expect("Serialization error");

            // let buffers =

            // let mut json_offset = json_string.len() as u32;
            // let glb = gltf::binary::Glb {
            //     header: gltf::binary::Header {
            //         magic: *b"glTF",
            //         version: 2,
            //         length: json_offset + buffer_length,
            //     },
            //     bin: Some(Cow::Owned(to_padded_byte_vector(triangle_vertices))),
            //     json: Cow::Owned(json_string.into_bytes()),
            // };
            // let writer = std::fs::File::create("triangle.glb").expect("I/O error");
            // glb.to_writer(writer).expect("glTF binary output error");
        }
    }
    // let yo = gltf_json::Buffer {
    //     byte_length
    // }

    println!("Done!");
}
