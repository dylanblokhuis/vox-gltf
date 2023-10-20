#![allow(unused)]

use vox_gltf::convert_vox_to_gltf;
use vox_gltf::dot_vox::load;

use criterion::{black_box, criterion_group, criterion_main, Criterion};

pub fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("vox_to_gltf", |b| b.iter(|| {
      let vox = load("./monu1.vox").unwrap();
      let output = vox_gltf::GltfOutput::Glb;
      convert_vox_to_gltf(black_box(vox), black_box(output));
    }));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
