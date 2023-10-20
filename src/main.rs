
use core::panic;
use std::fs;
use std::io::Write;
use std::path::Path;
use clap::Parser;

use vox_gltf::dot_vox::load;
use vox_gltf::{convert_vox_to_gltf, GltfOutput};


#[derive(Parser, Debug)]
#[command(author, version, long_about = None)]
struct Args {
    /// Path of .vox file to convert
    #[arg(id = "PATH")]
    path: String,

    /// Path to output file or directory, depending on output type. By default, the current working directory is used.
    #[arg(short, long)]
    destination: Option<String>,

    /// Output type, one of: seperate, glb, embedded
    #[arg(short, long, default_value = "glb")]
    output: String,
}

fn main() {
    let args = Args::parse();
    let vox = load(&args.path).expect("Failed to load vox file");
    let output = match args.output.as_str() {
        "seperate" => GltfOutput::GltfSeperate,
        "glb" => GltfOutput::Glb,
        "embedded" => GltfOutput::GltfEmbedded,
        _ => panic!("\nInvalid output type, must be one of: seperate, glb, embedded\nExample: `vox_gltf ./monu1.vox glb`\n"),
    };
    let gltf = convert_vox_to_gltf(vox, output);
    let destination = args.destination.unwrap_or(".".to_string());
    let path = Path::new(&destination);

    match output {
        GltfOutput::GltfSeperate => {
            // if not dir, error
            if !path.is_dir() {
                panic!("Destination must be a directory for `seperate` output");
            }

            let writer = fs::File::create(path.join("output.gltf")).expect("I/O error");
            gltf::json::serialize::to_writer_pretty(writer, &gltf.root).expect("Serialization error");

            let mut writer = fs::File::create(path.join("buffer0.bin")).expect("I/O error");
            writer
                .write_all(&gltf.buffer)
                .expect("I/O error");
        }
        GltfOutput::Glb => {
            if path.is_dir() {
                panic!("Destination must be a file for `glb` output");
            }

            let glb = gltf.into_glb();
            let writer = std::fs::File::create(path).expect("I/O error");
            glb.to_writer(writer).expect("glTF binary output error");
        }
        GltfOutput::GltfEmbedded => {
            todo!();
        }
    }
}
