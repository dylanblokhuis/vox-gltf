
use std::fs;
use std::io::Write;

use vox_gltf::dot_vox::load;
use vox_gltf::{convert_vox_to_gltf, GltfOutput};

fn main() {
    let vox = load("./monu1.vox").unwrap();
    let output = vox_gltf::GltfOutput::Glb;
    let gltf = convert_vox_to_gltf(vox, output);

    let _ = fs::remove_dir_all("output");
    let _ = fs::create_dir("output");

    match output {
        GltfOutput::GltfSeperate => {
            let writer = fs::File::create("output/output.gltf").expect("I/O error");
            gltf::json::serialize::to_writer_pretty(writer, &gltf.root).expect("Serialization error");

            let mut writer = fs::File::create("output/buffer0.bin").expect("I/O error");
            writer
                .write_all(&gltf.buffer)
                .expect("I/O error");
        }
        GltfOutput::Glb => {
            let glb = gltf.into_glb();
            let writer = std::fs::File::create("output/triangle.glb").expect("I/O error");
            glb.to_writer(writer).expect("glTF binary output error");
        }
        GltfOutput::GltfEmbedded => {
            todo!();
        }
    }
}
