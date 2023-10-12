use std::{collections::HashMap, fs, io::Write};

use block_mesh::{
    greedy_quads,
    ndshape::{ConstShape, ConstShape3u32},
    Axis, AxisPermutation, GreedyQuadsBuffer, MergeVoxel, OrientedBlockFace, QuadCoordinateConfig,
    Voxel, VoxelVisibility,
};
use dot_vox::load;
use slab::Slab;

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

#[repr(C, packed)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    color: [f32; 4],
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

fn align_to_multiple_of_four(n: &mut u32) {
    *n = (*n + 3) & !3;
}

fn to_padded_byte_vector<T>(vec: Vec<T>) -> Vec<u8> {
    let byte_length = vec.len() * std::mem::size_of::<T>();
    let byte_capacity = vec.capacity() * std::mem::size_of::<T>();
    let alloc = vec.into_boxed_slice();
    let ptr = Box::<[T]>::into_raw(alloc) as *mut u8;
    let mut new_vec = unsafe { Vec::from_raw_parts(ptr, byte_length, byte_capacity) };
    while new_vec.len() % 4 != 0 {
        new_vec.push(0); // pad to multiple of four bytes
    }
    new_vec
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

    // let num_indices = buffer.quads.num_quads() * 6;
    // let num_vertices = buffer.quads.num_quads() * 4;

    // let mut indices = Vec::with_capacity(num_indices);
    // let mut positions = Vec::with_capacity(num_vertices);
    // let mut normals = Vec::with_capacity(num_vertices);
    // let mut vertices = Vec::with_capacity(num_vertices);
    // let mut colors = Vec::with_capacity(num_vertices);

    let mut grouped_vertices: HashMap<u8, Vec<Vertex>> = HashMap::new();
    let mut grouped_indices: HashMap<u8, Vec<u32>> = HashMap::new();

    for (group, face) in buffer.quads.groups.into_iter().zip(faces.into_iter()) {
        for quad in group.into_iter() {
            let palette_index = voxels[ChunkShape::linearize(quad.minimum) as usize].0;
            let color = vox.palette[palette_index as usize];

            let vertices_entry = grouped_vertices.entry(palette_index).or_insert(Vec::new());
            let indices_entry = grouped_indices.entry(palette_index).or_insert(Vec::new());

            indices_entry.extend_from_slice(&face.quad_mesh_indices(vertices_entry.len() as u32));

            let mut vertices = Vec::with_capacity(4);
            for (position, normals) in face
                .quad_mesh_positions(&quad, 1.0)
                .iter()
                .zip(face.quad_mesh_normals())
            {
                vertices.push(Vertex {
                    position: *position,
                    normal: normals,
                    color: [
                        color.r as f32 / 255.0,
                        color.g as f32 / 255.0,
                        color.b as f32 / 255.0,
                        color.a as f32 / 255.0,
                    ],
                })
            }
            vertices_entry.extend_from_slice(&vertices);
        }
    }

    println!("{:?} {:?}", grouped_vertices.len(), grouped_indices.len());

    let output = Output::Standard;
    let mut accessors = Slab::<gltf_json::Accessor>::new();
    let mut buffer_views = Slab::<gltf_json::buffer::View>::new();
    let mut buffer = Vec::<u8>::new();

    fn create_primitive(
        vertices: Vec<Vertex>,
        indices: Vec<u32>,
        accessors: &mut Slab<gltf_json::Accessor>,
        buffer: &mut Vec<u8>,
        buffer_views: &mut Slab<gltf_json::buffer::View>,
        output: Output,
        offset: &mut u32,
    ) -> gltf_json::mesh::Primitive {
        let (min, max) = bounding_coords(&vertices);
        let vertex_buffer_length = (vertices.len() * std::mem::size_of::<Vertex>()) as u32;
        let indices_buffer_length = (indices.len() * std::mem::size_of::<u32>()) as u32;

        // let mut combined = Vec::with_capacity(vertex_buffer_length as usize + indices_buffer_length as usize);
        buffer.extend_from_slice(&to_padded_byte_vector(vertices.clone()));
        buffer.extend_from_slice(&to_padded_byte_vector(indices.clone()));
        
        // let buffer = buffers.insert((gltf_json::Buffer {
        //     byte_length: vertex_buffer_length + indices_buffer_length,
        //     extensions: Default::default(),
        //     extras: Default::default(),
        //     name: None,
        //     uri: if output == Output::Standard {
        //         Some(format!("buffer{}.bin", key))
        //     } else {
        //         None
        //     },
        // }, combined));

        let vertex_buffer_view = buffer_views.insert(gltf_json::buffer::View {
            buffer: gltf_json::Index::new(0),
            byte_length: vertex_buffer_length,
            byte_offset: Some(*offset),
            byte_stride: Some(std::mem::size_of::<Vertex>() as u32),
            extensions: Default::default(),
            extras: Default::default(),
            name: None,
            target: Some(gltf_json::validation::Checked::Valid(
                gltf_json::buffer::Target::ArrayBuffer,
            )),
        });
        *offset += vertex_buffer_length;
        let indices_buffer_view = buffer_views.insert(gltf_json::buffer::View {
            buffer: gltf_json::Index::new(0),
            byte_length: indices_buffer_length,
            byte_offset: Some(*offset),
            byte_stride: None,
            extensions: Default::default(),
            extras: Default::default(),
            name: None,
            target: Some(gltf_json::validation::Checked::Valid(
                gltf_json::buffer::Target::ElementArrayBuffer,
            )),
        });
        *offset += indices_buffer_length;

        let positions = accessors.insert(gltf_json::Accessor {
            buffer_view: Some(gltf_json::Index::new(vertex_buffer_view as u32)),
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
        });
        let normal = accessors.insert(gltf_json::Accessor {
            buffer_view: Some(gltf_json::Index::new(vertex_buffer_view as u32)),
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
        });

        let colors = accessors.insert(gltf_json::Accessor {
            buffer_view: Some(gltf_json::Index::new(vertex_buffer_view as u32)),
            byte_offset: Some((6 * std::mem::size_of::<f32>()) as u32),
            count: vertices.len() as u32,
            component_type: gltf_json::validation::Checked::Valid(
                gltf_json::accessor::GenericComponentType(gltf_json::accessor::ComponentType::F32),
            ),
            extensions: Default::default(),
            extras: Default::default(),
            type_: gltf_json::validation::Checked::Valid(gltf_json::accessor::Type::Vec4),
            min: None,
            max: None,
            name: None,
            normalized: false,
            sparse: None,
        });

        let indices_accessor = accessors.insert(gltf_json::Accessor {
            buffer_view: Some(gltf_json::Index::new(indices_buffer_view as u32)),
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
        });

        // let material = gltf_json::Material {
        //     pbr_metallic_roughness: gltf_json::material::PbrMetallicRoughness {
        //         base_color_factor: PbrBaseColorFactor()
        //         ..Default::default()
        //     },

        //     ..Default::default()
        // };

        gltf_json::mesh::Primitive {
            attributes: {
                let mut map = std::collections::BTreeMap::new();
                map.insert(
                    gltf_json::validation::Checked::Valid(gltf_json::mesh::Semantic::Positions),
                    gltf_json::Index::new(positions as u32),
                );
                map.insert(
                    gltf_json::validation::Checked::Valid(gltf_json::mesh::Semantic::Normals),
                    gltf_json::Index::new(normal as u32),
                );
                map.insert(
                    gltf_json::validation::Checked::Valid(gltf_json::mesh::Semantic::Colors(0)),
                    gltf_json::Index::new(colors as u32),
                );

                map
            },
            extensions: Default::default(),
            extras: Default::default(),
            indices: Some(gltf_json::Index::new(indices_accessor as u32)),
            material: None,
            mode: gltf_json::validation::Checked::Valid(gltf_json::mesh::Mode::Triangles),
            targets: None,
        }
    }

    let buffer_length = grouped_vertices
        .iter()
        .map(|(_, vertices)| vertices.len() * std::mem::size_of::<Vertex>())
        .sum::<usize>()
        + grouped_indices
            .iter()
            .map(|(_, indices)| indices.len() * std::mem::size_of::<u32>())
            .sum::<usize>();


    let uber_buffer = gltf_json::Buffer {
        byte_length: buffer_length as u32,
        extensions: Default::default(),
        extras: Default::default(),
        name: None,
        uri: if output == Output::Standard {
            Some("buffer0.bin".into())
        } else {
            None
        },
    };

    let mut offset = 0;

    let primitives = grouped_vertices
        .into_iter()
        .zip(grouped_indices.into_iter())
        .map(|((palette_index, vertices), (_, indices))| {
            create_primitive(vertices, indices, &mut accessors, &mut buffer, &mut buffer_views, output, &mut offset)
        })
        .collect::<Vec<_>>();

    let mesh = gltf_json::Mesh {
        extensions: Default::default(),
        extras: Default::default(),
        name: None,
        primitives: primitives,
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
        accessors: accessors.into_iter().map(|(_, v)| v).collect(),
        buffers: vec![uber_buffer],
        buffer_views: buffer_views.into_iter().map(|(_, v)| v).collect(),
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
            let _ = fs::remove_dir_all("output");
            let _ = fs::create_dir("output");

            let writer = fs::File::create("output/output.gltf").expect("I/O error");
            gltf_json::serialize::to_writer_pretty(writer, &root).expect("Serialization error");

            // let path = format!("output/{}", buffer.uri.as_ref().unwrap());
            let mut writer = fs::File::create("output/buffer0.bin").expect("I/O error");
            writer
                .write_all(&to_padded_byte_vector(buffer))
                .expect("I/O error");

            // let combined = vec![];

            // grouped_vertices.iter()

            // for (_, (buffer, data)) in buffers {
            //     let path = format!("output/{}", buffer.uri.as_ref().unwrap());
            //     let mut writer = fs::File::create(path).expect("I/O error");
            //     writer
            //         .write_all(&to_padded_byte_vector(data))
            //         .expect("I/O error");
            // }
        }
        Output::Binary => {
            let _ = fs::remove_dir_all("output");
            let _ = fs::create_dir("output");
            let json_string = gltf_json::serialize::to_string(&root).expect("Serialization error");

            let mut json_offset = json_string.len() as u32;
            align_to_multiple_of_four(&mut json_offset);

            // let mut combined = Vec::with_capacity(buffers.iter().map(|x| x.1 .1.len()).sum::<usize>());
            // for (_, (_, data)) in buffers {
            //     combined.extend_from_slice(&data);
            // }


            

            // let mut combined = Vec::with_capacity(vertex_buffer_length as usize + indices_buffer_length as usize);
            // combined.extend_from_slice(&to_padded_byte_vector(vertices));
            // combined.extend_from_slice(&to_padded_byte_vector(indices));

            // println!("{}", combined.len());

            let combined = to_padded_byte_vector(buffer);

            let glb = gltf::binary::Glb {
                header: gltf::binary::Header {
                    magic: *b"glTF",
                    version: 2,
                    length: json_offset + combined.len() as u32,
                },
                bin: Some(std::borrow::Cow::Owned(combined)),
                json: std::borrow::Cow::Owned(json_string.into_bytes()),
            };
            let writer = std::fs::File::create("output/triangle.glb").expect("I/O error");
            glb.to_writer(writer).expect("glTF binary output error");
        }
    }

    println!("Done!");
}
