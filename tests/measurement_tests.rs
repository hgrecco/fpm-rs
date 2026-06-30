use std::{
    fs::File,
    io::BufWriter,
    path::PathBuf,
    sync::{Arc, Barrier},
    thread,
};

use fpm_rs::measurements::{
    FrameMetadata, ImagePreprocessingConfig, LazyMeasurementStack, ManifestFrame, ManifestImageSet,
    MeasurementManifest, MeasurementRead, MeasurementStack,
};
use image::{GrayImage, ImageBuffer, Luma};

fn assert_measurement_read<M: MeasurementRead>(
    measurements: &M,
    expected_first: &[f64],
    expected_second: &[f64],
) {
    assert_eq!(measurements.frame_count(), 2);
    assert_eq!(measurements.image_shape(), (1, 2));
    assert_eq!(measurements.frame_len(), 2);
    let first = measurements.frame(0).unwrap();
    let second = measurements.frame(1).unwrap();
    assert_eq!(&*first, expected_first);
    assert_eq!(&*second, expected_second);
    assert_eq!(measurements.frame_weight(0).unwrap(), 1.0);
    assert!(measurements.frame_mask(0).unwrap().is_none());
    assert_eq!(measurements.frame_metadata().len(), 2);
    measurements.validate().unwrap();
}

#[test]
fn in_memory_stack_implements_read_only_measurement_access() {
    let stack = MeasurementStack::from_vec(vec![1.0, 2.0, 3.0, 4.0], (1, 2), Vec::new())
        .unwrap();
    assert_measurement_read(&stack, &[1.0, 2.0], &[3.0, 4.0]);
}

#[test]
fn preprocessing_applies_dark_background_flat_and_exposure_in_order() {
    let shape = (1, 3);
    let mut first = FrameMetadata::new(0);
    first.exposure_time = 2.0;
    let mut second = FrameMetadata::new(1);
    second.exposure_time = 4.0;
    // raw = dark + background + flat * exposure * desired_signal
    let stack = MeasurementStack::from_vec(
        vec![15.0, 7.0, -1.0, 27.0, 11.0, -5.0],
        shape,
        vec![first, second],
    )
    .unwrap()
    .with_dark_frame(vec![1.0; 3])
    .unwrap()
    .with_background(vec![2.0; 3])
    .unwrap()
    .with_flat_field(vec![2.0; 3])
    .unwrap()
    .normalize_exposure()
    .clamp_negative()
    .apply_preprocessing()
    .unwrap();
    assert_eq!(stack.frame(0).unwrap(), &[3.0, 1.0, 0.0]);
    assert_eq!(stack.frame(1).unwrap(), &[3.0, 1.0, 0.0]);
}

#[test]
fn metadata_and_corrections_are_validated() {
    let mut metadata = FrameMetadata::new(0);
    metadata.weight = f64::NAN;
    assert!(MeasurementStack::from_vec(vec![1.0], (1, 1), vec![metadata]).is_err());

    let stack = MeasurementStack::from_vec(vec![1.0, 2.0], (1, 2), Vec::new()).unwrap();
    assert!(stack.clone().with_flat_field(vec![1.0, 0.0]).is_err());
    assert!(stack.with_masks(vec![1, 0, 1]).is_err());
}

#[test]
fn measurement_deserialization_preserves_and_validates_private_invariants() {
    let stack = MeasurementStack::from_vec(vec![1.0, 2.0, 3.0, 4.0], (1, 2), Vec::new())
        .unwrap()
        .with_masks(vec![1, 0])
        .unwrap();
    let encoded = serde_json::to_string(&stack).unwrap();
    let decoded: MeasurementStack = serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded.image_shape(), stack.image_shape());
    assert_eq!(decoded.frame_count(), stack.frame_count());
    assert_eq!(decoded.as_slice(), stack.as_slice());
    assert_eq!(decoded.frame_mask(1).unwrap(), stack.frame_mask(1).unwrap());

    let mut inconsistent = serde_json::to_value(&stack).unwrap();
    inconsistent["frames"] = serde_json::json!(usize::MAX);
    assert!(serde_json::from_value::<MeasurementStack>(inconsistent).is_err());

    let mut overflowing = serde_json::to_value(&stack).unwrap();
    overflowing["image_shape"] = serde_json::json!([usize::MAX, 2]);
    assert!(serde_json::from_value::<MeasurementStack>(overflowing).is_err());
}

#[test]
fn masks_can_be_broadcast_or_stored_per_frame() {
    let broadcast = MeasurementStack::from_vec(vec![1.0; 8], (2, 2), Vec::new())
        .unwrap()
        .with_masks(vec![1, 0, 1, 1])
        .unwrap();
    assert_eq!(broadcast.frame_mask(0).unwrap().unwrap(), &[1, 0, 1, 1]);
    assert_eq!(broadcast.frame_mask(1).unwrap().unwrap(), &[1, 0, 1, 1]);

    let per_frame = MeasurementStack::from_vec(vec![1.0; 8], (2, 2), Vec::new())
        .unwrap()
        .with_masks(vec![1, 1, 1, 1, 0, 1, 1, 1])
        .unwrap();
    assert_eq!(per_frame.frame_mask(0).unwrap().unwrap(), &[1, 1, 1, 1]);
    assert_eq!(per_frame.frame_mask(1).unwrap().unwrap(), &[0, 1, 1, 1]);
}

#[test]
fn lazy_stack_decodes_frames_on_demand_and_materializes() {
    let directory = tempfile::tempdir().unwrap();
    let first_path = directory.path().join("lazy0.png");
    let second_path = directory.path().join("lazy1.png");
    GrayImage::from_vec(2, 2, vec![1, 2, 3, 4])
        .unwrap()
        .save(&first_path)
        .unwrap();
    GrayImage::from_vec(2, 2, vec![5, 6, 7, 8])
        .unwrap()
        .save(&second_path)
        .unwrap();
    let stack =
        LazyMeasurementStack::from_image_files(&[first_path.clone(), second_path], Vec::new())
            .unwrap();
    assert_eq!(stack.cached_frame_count(), 0);
    assert_eq!(stack.frame(0).unwrap().as_slice(), &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(stack.cached_frame_count(), 1);

    std::fs::remove_file(first_path).unwrap();
    assert_eq!(stack.frame(0).unwrap().as_slice(), &[1.0, 2.0, 3.0, 4.0]);
    let materialized = stack.materialize().unwrap();
    assert_eq!(stack.cached_frame_count(), 1);
    assert_eq!(materialized.frame(1).unwrap(), &[5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn lazy_stack_implements_read_only_measurement_access() {
    let directory = tempfile::tempdir().unwrap();
    let first = directory.path().join("read0.png");
    let second = directory.path().join("read1.png");
    GrayImage::from_vec(2, 1, vec![1, 2])
        .unwrap()
        .save(&first)
        .unwrap();
    GrayImage::from_vec(2, 1, vec![3, 4])
        .unwrap()
        .save(&second)
        .unwrap();
    let stack = LazyMeasurementStack::from_image_files(&[first, second], Vec::new()).unwrap();

    assert_measurement_read(&stack, &[1.0, 2.0], &[3.0, 4.0]);
}

#[test]
fn lazy_stack_preprocesses_each_frame_during_decode() {
    let directory = tempfile::tempdir().unwrap();
    let first_path = directory.path().join("processed0.png");
    let second_path = directory.path().join("processed1.png");
    GrayImage::from_vec(3, 1, vec![15, 7, 1])
        .unwrap()
        .save(&first_path)
        .unwrap();
    GrayImage::from_vec(3, 1, vec![27, 11, 1])
        .unwrap()
        .save(&second_path)
        .unwrap();
    let mut first = FrameMetadata::new(0);
    first.exposure_time = 2.0;
    let mut second = FrameMetadata::new(1);
    second.exposure_time = 4.0;

    let stack =
        LazyMeasurementStack::from_image_files(&[first_path, second_path], vec![first, second])
            .unwrap()
            .with_dark_frame(vec![1.0; 3])
            .unwrap()
            .with_background(vec![2.0; 6])
            .unwrap()
            .with_flat_field(vec![2.0; 3])
            .unwrap()
            .normalize_exposure()
            .clamp_negative();

    assert_eq!(stack.cached_frame_count(), 0);
    assert_eq!(stack.frame(0).unwrap().as_slice(), &[3.0, 1.0, 0.0]);
    assert_eq!(stack.frame(1).unwrap().as_slice(), &[3.0, 1.0, 0.0]);
    let materialized = stack.materialize().unwrap();
    assert_eq!(materialized.frame(0).unwrap(), &[3.0, 1.0, 0.0]);
    assert_eq!(materialized.frame(1).unwrap(), &[3.0, 1.0, 0.0]);
}

#[test]
fn lazy_cache_enforces_byte_limit_under_concurrent_reads() {
    let directory = tempfile::tempdir().unwrap();
    let paths: Vec<_> = (0..4)
        .map(|index| {
            let path = directory.path().join(format!("cache{index}.png"));
            GrayImage::from_vec(2, 2, vec![index as u8; 4])
                .unwrap()
                .save(&path)
                .unwrap();
            path
        })
        .collect();
    let frame_bytes = 4 * std::mem::size_of::<f64>();
    assert!(
        LazyMeasurementStack::from_image_files(&paths, Vec::new())
            .unwrap()
            .with_cache_byte_capacity(frame_bytes - 1)
            .is_err()
    );
    let stack = Arc::new(
        LazyMeasurementStack::from_image_files(&paths, Vec::new())
            .unwrap()
            .with_cache_capacity(4)
            .unwrap()
            .with_cache_byte_capacity(frame_bytes * 2)
            .unwrap(),
    );
    let barrier = Arc::new(Barrier::new(8));
    let threads: Vec<_> = (0..8)
        .map(|worker| {
            let stack = Arc::clone(&stack);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                for offset in 0..16 {
                    let index = (worker + offset) % 4;
                    let frame = stack.frame(index).unwrap();
                    assert_eq!(frame.as_slice(), &[index as f64; 4]);
                }
            })
        })
        .collect();
    for worker in threads {
        worker.join().unwrap();
    }
    assert!(stack.cached_frame_count() <= 2);
    assert!(stack.cached_byte_count() <= frame_bytes * 2);
}

#[test]
fn lazy_byte_cache_evicts_the_least_recently_used_frame() {
    let directory = tempfile::tempdir().unwrap();
    let paths: Vec<_> = (0..3)
        .map(|index| {
            let path = directory.path().join(format!("evict{index}.png"));
            GrayImage::from_vec(2, 2, vec![index as u8; 4])
                .unwrap()
                .save(&path)
                .unwrap();
            path
        })
        .collect();
    let frame_bytes = 4 * std::mem::size_of::<f64>();
    let stack = LazyMeasurementStack::from_image_files(&paths, Vec::new())
        .unwrap()
        .with_cache_capacity(3)
        .unwrap()
        .with_cache_byte_capacity(frame_bytes * 2)
        .unwrap();

    stack.frame(0).unwrap();
    stack.frame(1).unwrap();
    stack.frame(0).unwrap();
    stack.frame(2).unwrap();
    assert_eq!(stack.cached_frame_count(), 2);
    assert_eq!(stack.cached_byte_count(), frame_bytes * 2);

    for path in &paths {
        std::fs::remove_file(path).unwrap();
    }
    assert_eq!(stack.frame(0).unwrap().as_slice(), &[0.0; 4]);
    assert_eq!(stack.frame(2).unwrap().as_slice(), &[2.0; 4]);
    assert!(stack.frame(1).is_err());
}

#[test]
fn image_stack_loader_preserves_native_8_and_16_bit_counts() {
    let directory = tempfile::tempdir().unwrap();
    let eight_bit_path = directory.path().join("frame8.png");
    let sixteen_bit_path = directory.path().join("frame16.png");
    GrayImage::from_vec(2, 2, vec![0, 1, 127, 255])
        .unwrap()
        .save(&eight_bit_path)
        .unwrap();
    let sixteen_bit: ImageBuffer<Luma<u16>, Vec<u16>> =
        ImageBuffer::from_vec(2, 2, vec![0, 256, 4095, 65535]).unwrap();
    sixteen_bit.save(&sixteen_bit_path).unwrap();

    let eight_bit = MeasurementStack::from_image_files(&[&eight_bit_path], Vec::new()).unwrap();
    assert_eq!(eight_bit.frame(0).unwrap(), &[0.0, 1.0, 127.0, 255.0]);
    assert!(
        eight_bit.frame_metadata[0]
            .label
            .as_ref()
            .unwrap()
            .ends_with("frame8.png")
    );

    let sixteen_bit = MeasurementStack::from_image_files(&[&sixteen_bit_path], Vec::new()).unwrap();
    assert_eq!(
        sixteen_bit.frame(0).unwrap(),
        &[0.0, 256.0, 4095.0, 65535.0]
    );
}

#[test]
fn image_stack_loader_rejects_empty_and_inconsistent_stacks() {
    let no_paths: [&std::path::Path; 0] = [];
    assert!(MeasurementStack::from_image_files(&no_paths, Vec::new()).is_err());

    let directory = tempfile::tempdir().unwrap();
    let first = directory.path().join("first.png");
    let second = directory.path().join("second.png");
    GrayImage::new(2, 2).save(&first).unwrap();
    GrayImage::new(3, 2).save(&second).unwrap();
    assert!(MeasurementStack::from_image_files(&[first, second], Vec::new()).is_err());
}

#[test]
fn image_stack_loader_preserves_native_tiff_counts() {
    let directory = tempfile::tempdir().unwrap();
    let eight_bit_path = directory.path().join("frame8.tiff");
    let sixteen_bit_path = directory.path().join("frame16.tiff");
    GrayImage::from_vec(2, 2, vec![0, 1, 127, 255])
        .unwrap()
        .save(&eight_bit_path)
        .unwrap();
    let sixteen_bit: ImageBuffer<Luma<u16>, Vec<u16>> =
        ImageBuffer::from_vec(2, 2, vec![0, 256, 4095, 65535]).unwrap();
    sixteen_bit.save(&sixteen_bit_path).unwrap();

    let mixed = MeasurementStack::from_image_files(
        &[eight_bit_path.as_path(), sixteen_bit_path.as_path()],
        Vec::new(),
    )
    .unwrap();
    assert_eq!(mixed.frame_count(), 2);
    assert_eq!(mixed.frame(0).unwrap(), &[0.0, 1.0, 127.0, 255.0]);
    assert_eq!(mixed.frame(1).unwrap(), &[0.0, 256.0, 4095.0, 65535.0]);

    let eight_bit = MeasurementStack::from_image_files(&[eight_bit_path], Vec::new()).unwrap();
    assert_eq!(eight_bit.frame(0).unwrap(), &[0.0, 1.0, 127.0, 255.0]);
    let sixteen_bit = MeasurementStack::from_image_files(&[sixteen_bit_path], Vec::new()).unwrap();
    assert_eq!(
        sixteen_bit.frame(0).unwrap(),
        &[0.0, 256.0, 4095.0, 65535.0]
    );
}

#[test]
fn multipage_tiff_loader_reads_each_page_as_a_frame() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("stack.tiff");
    let writer = BufWriter::new(File::create(&path).unwrap());
    let mut encoder = tiff::encoder::TiffEncoder::new(writer).unwrap();
    encoder
        .write_image::<tiff::encoder::colortype::Gray16>(2, 2, &[1, 2, 3, 4])
        .unwrap();
    encoder
        .write_image::<tiff::encoder::colortype::Gray16>(2, 2, &[100, 200, 300, 400])
        .unwrap();
    drop(encoder);

    let stack = MeasurementStack::from_tiff_stack(&path, Vec::new()).unwrap();
    assert_eq!(stack.frame_count(), 2);
    assert_eq!(stack.image_shape(), (2, 2));
    assert_eq!(stack.frame(0).unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(stack.frame(1).unwrap(), &[100.0, 200.0, 300.0, 400.0]);
    assert!(
        stack.frame_metadata[1]
            .label
            .as_deref()
            .unwrap()
            .ends_with("stack.tiff#page=1")
    );

    let lazy = LazyMeasurementStack::from_tiff_stack(&path, Vec::new()).unwrap();
    assert_eq!(lazy.frame_count(), 2);
    assert_eq!(lazy.image_shape(), (2, 2));
    assert_eq!(lazy.cached_frame_count(), 0);
    assert_eq!(
        lazy.frame(1).unwrap().as_slice(),
        &[100.0, 200.0, 300.0, 400.0]
    );
    assert_eq!(lazy.cached_frame_count(), 1);
    assert!(
        lazy.frame_metadata()[1]
            .label
            .as_deref()
            .unwrap()
            .ends_with("stack.tiff#page=1")
    );
}

#[test]
fn manifest_loads_relative_images_metadata_and_preprocessing() {
    let directory = tempfile::tempdir().unwrap();
    let save_u16 = |name: &str, values: Vec<u16>| {
        let path = directory.path().join(name);
        let image: ImageBuffer<Luma<u16>, Vec<u16>> = ImageBuffer::from_vec(2, 1, values).unwrap();
        image.save(&path).unwrap();
    };
    save_u16("frame0.tiff", vec![20, 10]);
    save_u16("frame1.tiff", vec![30, 14]);
    save_u16("dark.tiff", vec![2, 2]);
    save_u16("flat.tiff", vec![2, 2]);
    save_u16("background.tiff", vec![4, 4]);
    save_u16("mask.tiff", vec![1, 0]);

    let mut first = ManifestFrame::new("frame0.tiff");
    first.illumination_index = Some(7);
    first.exposure_time = 2.0;
    first.weight = 0.5;
    first.label = Some("brightfield".into());
    let mut second = ManifestFrame::new("frame1.tiff");
    second.exposure_time = 4.0;
    let mut manifest = MeasurementManifest::new(vec![first, second]);
    manifest.dark_frame = Some(PathBuf::from("dark.tiff"));
    manifest.flat_field = Some(PathBuf::from("flat.tiff"));
    manifest.background = Some(ManifestImageSet::Single(PathBuf::from("background.tiff")));
    manifest.mask = Some(ManifestImageSet::Single(PathBuf::from("mask.tiff")));
    manifest.preprocessing = ImagePreprocessingConfig {
        subtract_dark: true,
        divide_flat_field: true,
        normalize_exposure: true,
        subtract_background: true,
        clamp_negative: true,
    };
    let manifest_path = directory.path().join("measurements.json");
    manifest.save(&manifest_path).unwrap();

    let stack = MeasurementStack::from_manifest(&manifest_path).unwrap();
    assert_eq!(stack.frame(0).unwrap(), &[20.0, 10.0]);
    assert_eq!(stack.frame_metadata[0].illumination_index, Some(7));
    assert_eq!(stack.frame_metadata[0].weight, 0.5);
    assert_eq!(
        stack.frame_metadata[0].label.as_deref(),
        Some("brightfield")
    );
    assert_eq!(
        stack.frame_metadata[1].label.as_deref(),
        Some("frame1.tiff")
    );
    assert_eq!(stack.frame_mask(0).unwrap().unwrap(), &[1, 0]);

    let processed = stack.apply_preprocessing().unwrap();
    assert_eq!(processed.frame(0).unwrap(), &[3.5, 1.0]);
    assert_eq!(processed.frame(1).unwrap(), &[3.0, 1.0]);

    let lazy = LazyMeasurementStack::from_manifest(&manifest_path).unwrap();
    assert_eq!(lazy.cached_frame_count(), 0);
    assert_eq!(lazy.frame(0).unwrap().as_slice(), &[3.5, 1.0]);
    assert_eq!(lazy.frame(1).unwrap().as_slice(), &[3.0, 1.0]);
    assert_eq!(lazy.frame_mask(0).unwrap().unwrap(), &[1, 0]);
}

#[test]
fn manifest_rejects_unknown_fields() {
    let directory = tempfile::tempdir().unwrap();
    let manifest_path = directory.path().join("invalid.json");
    std::fs::write(
        &manifest_path,
        r#"{"frames":[],"preprocesing":{"normalize_exposure":true}}"#,
    )
    .unwrap();
    assert!(MeasurementManifest::load(manifest_path).is_err());
}
