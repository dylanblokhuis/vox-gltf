use std::collections::HashMap;

use block_mesh::{
    greedy_quads,
    ndshape::{ConstShape, ConstShape3u32},
    Axis, AxisPermutation, GreedyQuadsBuffer, MergeVoxel, OrientedBlockFace, QuadCoordinateConfig,
    Voxel, VoxelVisibility,
};
use dot_vox::DotVoxData;
use gltf::{json::{self, extensions::material::IndexOfRefraction}, Glb};
use gltf::json::{extensions::material::TransmissionFactor, material::PbrBaseColorFactor};
use slab::Slab;

pub mod dot_vox {
    pub use dot_vox::*;
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct BoolVoxel {
    palette_index: u8,
    visibility: VoxelVisibility,
}

pub const EMPTY_VOXEL: BoolVoxel = BoolVoxel {
    palette_index: 255,
    visibility: VoxelVisibility::Empty,
};

impl Voxel for BoolVoxel {
    fn get_visibility(&self) -> VoxelVisibility {
        self.visibility
    }
}

impl MergeVoxel for BoolVoxel {
    type MergeValue = Self;
    fn merge_value(&self) -> Self::MergeValue {
        *self
    }
}

#[repr(C, align(4))]
#[derive(Clone, Copy, Debug)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    color: [f32; 4],
    uv: [f32; 2],
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

pub struct GltfData {
    pub root: gltf::json::Root,
    pub buffer: Vec<u8>,
}

impl GltfData {
    pub fn into_glb(self) -> Glb<'static> {
        let json_string = gltf::json::serialize::to_string(&self.root).expect("Serialization error");

        let mut json_offset = json_string.len() as u32;
        align_to_multiple_of_four(&mut json_offset);

        // let combined = (buffer);
        let combined = to_padded_byte_vector(self.buffer);

        gltf::binary::Glb {
            header: gltf::binary::Header {
                magic: *b"glTF",
                version: 2,
                length: json_offset + combined.len() as u32,
            },
            bin: Some(std::borrow::Cow::Owned(combined)),
            json: std::borrow::Cow::Owned(json_string.into_bytes()),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GltfOutput {
    GltfSeperate,
    GltfEmbedded,
    Glb,
}

pub fn convert_vox_to_gltf(vox: DotVoxData, output: GltfOutput) -> GltfData {
    let model = &vox.models[0];

    log::info!("Voxel model size: {:?}", model.size);
    let mut voxels = vec![EMPTY_VOXEL; ChunkShape::SIZE as usize];
    log::debug!("{:?} {}", voxels.len(), model.voxels.len());

    for voxel in &model.voxels {
        let x = voxel.x as u32 + 1;
        let y = voxel.y as u32 + 1;
        let z = voxel.z as u32 + 1;

        let material = &vox.materials[voxel.i as usize];

        voxels[(x + z * MAX_SIZE + y * MAX_SIZE * MAX_SIZE) as usize] = BoolVoxel {
            palette_index: voxel.i,
            visibility: if material.transparency().unwrap_or(0.0) > 0.0 {
                VoxelVisibility::Translucent
            } else {
                VoxelVisibility::Opaque
            },
        }
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

    let mut grouped_vertices: HashMap<u8, Vec<Vertex>> = HashMap::new();
    let mut grouped_indices: HashMap<u8, Vec<u32>> = HashMap::new();

    for (group, face) in buffer.quads.groups.into_iter().zip(faces.into_iter()) {
        for quad in group.into_iter() {
            let palette_index = voxels[ChunkShape::linearize(quad.minimum) as usize].palette_index;
            let color = vox.palette[palette_index as usize];

            let vertices_entry = grouped_vertices.entry(palette_index).or_default();
            let indices_entry = grouped_indices.entry(palette_index).or_default();

            log::debug!(
                "Adding quad. Current vertex count: {}",
                vertices_entry.len()
            );
            log::debug!(
                "Indices being added: {:?}",
                &face.quad_mesh_indices(vertices_entry.len() as u32)
            );

            indices_entry.extend_from_slice(&face.quad_mesh_indices(vertices_entry.len() as u32));

            let mut vertices = Vec::with_capacity(4);
            for ((position, normals), uv) in face
                .quad_mesh_positions(&quad, 1.0)
                .iter()
                .zip(face.quad_mesh_normals())
                .zip(face.tex_coords(RIGHT_HANDED_Z_UP_CONFIG.u_flip_face, false, &quad))
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
                    uv,
                })
            }
            vertices_entry.extend_from_slice(&vertices);
            assert!(vertices.len() == 4, "Expected 4 vertices for a quad!");
        }
    }

    for (palette_index, vertices) in &grouped_vertices {
        let indices = &grouped_indices[palette_index];
        let max_index = vertices.len() as u32 - 1;
        for &index in indices {
            assert!(
                index <= max_index,
                "Index out of bounds for palette_index {}: Index {}, Max allowed: {}",
                palette_index,
                index,
                max_index
            );
        }
    }

    let mut accessors = Slab::<json::Accessor>::new();
    let mut buffer_views = Slab::<json::buffer::View>::new();
    let mut materials = Slab::<json::Material>::new();
    let mut buffer = Vec::<u8>::new();
    let mut offset = 0;

    let mut create_primitive =
        |palette_index: u8, vertices: Vec<Vertex>, indices: Vec<u32>| -> json::mesh::Primitive {
            let (min, max) = bounding_coords(&vertices);
            let vertex_buffer_length = (vertices.len() * std::mem::size_of::<Vertex>()) as u32;
            let indices_buffer_length = (indices.len() * std::mem::size_of::<u32>()) as u32;

            // let mut combined = Vec::with_capacity(vertex_buffer_length as usize + indices_buffer_length as usize);
            buffer.extend_from_slice(&to_padded_byte_vector(vertices.clone()));
            buffer.extend_from_slice(&to_padded_byte_vector(indices.clone()));

            let vertex_buffer_view = buffer_views.insert(json::buffer::View {
                buffer: json::Index::new(0),
                byte_length: vertex_buffer_length,
                byte_offset: Some(offset),
                byte_stride: Some(std::mem::size_of::<Vertex>() as u32),
                extensions: Default::default(),
                extras: Default::default(),
                name: None,
                target: Some(json::validation::Checked::Valid(
                    json::buffer::Target::ArrayBuffer,
                )),
            });
            offset += vertex_buffer_length;
            let indices_buffer_view = buffer_views.insert(json::buffer::View {
                buffer: json::Index::new(0),
                byte_length: indices_buffer_length,
                byte_offset: Some(offset),
                byte_stride: None,
                extensions: Default::default(),
                extras: Default::default(),
                name: None,
                target: Some(json::validation::Checked::Valid(
                    json::buffer::Target::ElementArrayBuffer,
                )),
            });
            offset += indices_buffer_length;

            let positions = accessors.insert(json::Accessor {
                buffer_view: Some(json::Index::new(vertex_buffer_view as u32)),
                byte_offset: Some(0),
                count: vertices.len() as u32,
                component_type: json::validation::Checked::Valid(
                    json::accessor::GenericComponentType(json::accessor::ComponentType::F32),
                ),
                extensions: Default::default(),
                extras: Default::default(),
                type_: json::validation::Checked::Valid(json::accessor::Type::Vec3),
                min: Some(json::Value::from(Vec::from(min))),
                max: Some(json::Value::from(Vec::from(max))),
                name: None,
                normalized: false,
                sparse: None,
            });
            let normal = accessors.insert(json::Accessor {
                buffer_view: Some(json::Index::new(vertex_buffer_view as u32)),
                byte_offset: Some((3 * std::mem::size_of::<f32>()) as u32),
                count: vertices.len() as u32,
                component_type: json::validation::Checked::Valid(
                    json::accessor::GenericComponentType(json::accessor::ComponentType::F32),
                ),
                extensions: Default::default(),
                extras: Default::default(),
                type_: json::validation::Checked::Valid(json::accessor::Type::Vec3),
                min: None,
                max: None,
                name: None,
                normalized: false,
                sparse: None,
            });

            let colors = accessors.insert(json::Accessor {
                buffer_view: Some(json::Index::new(vertex_buffer_view as u32)),
                byte_offset: Some((6 * std::mem::size_of::<f32>()) as u32),
                count: vertices.len() as u32,
                component_type: json::validation::Checked::Valid(
                    json::accessor::GenericComponentType(json::accessor::ComponentType::F32),
                ),
                extensions: Default::default(),
                extras: Default::default(),
                type_: json::validation::Checked::Valid(json::accessor::Type::Vec4),
                min: None,
                max: None,
                name: None,
                normalized: false,
                sparse: None,
            });

            let uv = accessors.insert(json::Accessor {
                buffer_view: Some(json::Index::new(vertex_buffer_view as u32)),
                byte_offset: Some((10 * std::mem::size_of::<f32>()) as u32),
                count: vertices.len() as u32,
                component_type: json::validation::Checked::Valid(
                    json::accessor::GenericComponentType(json::accessor::ComponentType::F32),
                ),
                extensions: Default::default(),
                extras: Default::default(),
                type_: json::validation::Checked::Valid(json::accessor::Type::Vec2),
                min: None,
                max: None,
                name: None,
                normalized: false,
                sparse: None,
            });

            let indices_accessor = accessors.insert(json::Accessor {
                buffer_view: Some(json::Index::new(indices_buffer_view as u32)),
                byte_offset: Some(0),
                count: indices.len() as u32,
                component_type: json::validation::Checked::Valid(
                    json::accessor::GenericComponentType(json::accessor::ComponentType::U32),
                ),
                extensions: Default::default(),
                extras: Default::default(),
                type_: json::validation::Checked::Valid(json::accessor::Type::Scalar),
                min: None,
                max: None,
                name: None,
                normalized: false,
                sparse: None,
            });

            let vox_material = &vox.materials[palette_index as usize];
            let vox_color = vox.palette[palette_index as usize];

            log::debug!("Material type: {:?}", vox_material);

            let material = materials.insert(json::Material {
                pbr_metallic_roughness: json::material::PbrMetallicRoughness {
                    base_color_factor: PbrBaseColorFactor([
                        vox_color.r as f32 / 255.0,
                        vox_color.g as f32 / 255.0,
                        vox_color.b as f32 / 255.0,
                        vox_color.a as f32 / 255.0,
                    ]),
                    metallic_factor: json::material::StrengthFactor(
                        vox_material.metalness().unwrap_or(0.0),
                    ),
                    roughness_factor: json::material::StrengthFactor(
                        vox_material.roughness().unwrap_or(0.0),
                    ),

                    ..Default::default()
                },
                emissive_factor: if vox_material.material_type() == Some("_emit") {
                    json::material::EmissiveFactor([
                        vox_color.r as f32 / 255.0,
                        vox_color.g as f32 / 255.0,
                        vox_color.b as f32 / 255.0,
                    ])
                } else {
                    json::material::EmissiveFactor([0.0, 0.0, 0.0])
                },

                extensions: Some(json::extensions::material::Material {
                    emissive_strength: if vox_material.material_type() == Some("_emit") {
                        Some(json::extensions::material::EmissiveStrength {
                            emissive_strength: json::extensions::material::EmissiveStrengthFactor(
                                (vox_material.emission().unwrap() * 10.0)
                                    * vox_material.radiant_flux().unwrap(),
                            ),
                        })
                    } else {
                        None
                    },
                    ior: Some(json::extensions::material::Ior {
                        ior: IndexOfRefraction(1.0 + vox_material.refractive_index().unwrap()),
                        ..Default::default()
                    }),
                    transmission: if vox_material.material_type() == Some("_glass") {
                        Some(json::extensions::material::Transmission {
                            transmission_factor: TransmissionFactor(
                                vox_material.transparency().unwrap_or(0.0),
                            ),
                            ..Default::default()
                        })
                    } else {
                        None
                    },
                }),
                ..Default::default()
            });

            json::mesh::Primitive {
                attributes: {
                    let mut map = std::collections::BTreeMap::new();
                    map.insert(
                        json::validation::Checked::Valid(json::mesh::Semantic::Positions),
                        json::Index::new(positions as u32),
                    );
                    map.insert(
                        json::validation::Checked::Valid(json::mesh::Semantic::Normals),
                        json::Index::new(normal as u32),
                    );
                    map.insert(
                        json::validation::Checked::Valid(json::mesh::Semantic::Colors(0)),
                        json::Index::new(colors as u32),
                    );
                    map.insert(
                        json::validation::Checked::Valid(json::mesh::Semantic::TexCoords(0)),
                        json::Index::new(uv as u32),
                    );

                    map
                },
                extensions: Default::default(),
                extras: Default::default(),
                indices: Some(json::Index::new(indices_accessor as u32)),
                material: Some(json::Index::new(material as u32)),
                mode: json::validation::Checked::Valid(json::mesh::Mode::Triangles),
                targets: None,
            }
        };

    let primitives = grouped_vertices
        .keys()
        .filter_map(|palette_index| {
            let vertices = grouped_vertices.get(palette_index)?;
            let indices = grouped_indices.get(palette_index)?;
            Some(create_primitive(
                *palette_index,
                vertices.clone(),
                indices.clone(),
            ))
        })
        .collect::<Vec<_>>();

    log::debug!("Buffer to be written: {} bytes", buffer.len());

    let uber_buffer = json::Buffer {
        byte_length: buffer.len() as u32,
        extensions: Default::default(),
        extras: Default::default(),
        name: None,
        uri: if output == GltfOutput::GltfSeperate {
            Some("buffer0.bin".into())
        } else {
            None
        },
    };

    let mesh = json::Mesh {
        extensions: Default::default(),
        extras: Default::default(),
        name: None,
        primitives,
        weights: None,
    };

    let node = json::Node {
        camera: None,
        children: None,
        extensions: Default::default(),
        extras: Default::default(),
        matrix: None,
        mesh: Some(json::Index::new(0)),
        name: None,
        rotation: None,
        scale: None,
        translation: None,
        skin: None,
        weights: None,
    };

    log::info!("Gltf root node with: {} accessors", accessors.len());

    GltfData {
        root: json::Root {
            accessors: accessors.into_iter().map(|(_, v)| v).collect(),
            buffers: vec![uber_buffer],
            buffer_views: buffer_views.into_iter().map(|(_, v)| v).collect(),
            meshes: vec![mesh],
            nodes: vec![node],
            scenes: vec![json::Scene {
                extensions: Default::default(),
                extras: Default::default(),
                name: None,
                nodes: vec![json::Index::new(0)],
            }],
            materials: materials.into_iter().map(|(_, v)| v).collect(),
            extensions_used: vec![
                "KHR_materials_emissive_strength".into(),
                "KHR_materials_ior".into(),
                "KHR_materials_transmission".into(),
            ],
            // extensions_required: vec!["KHR_materials_emissive_strength".into()],
            ..Default::default()
        },
        buffer: to_padded_byte_vector(buffer),
    }
}
