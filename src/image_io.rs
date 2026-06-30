use std::{fs::File, io::BufReader, path::Path};

use image::{ColorType, DynamicImage, ImageDecoder, ImageReader};

use crate::{Array2, Result, error::Error};

#[derive(Clone, Copy, Debug)]
pub(crate) enum GrayscaleScaling {
    NativeCounts,
    Unit,
}

pub(crate) fn grayscale_image_shape(path: impl AsRef<Path>) -> Result<(usize, usize)> {
    let path = path.as_ref();
    let decoder = ImageReader::open(path)?
        .with_guessed_format()?
        .into_decoder()?;
    if !matches!(decoder.color_type(), ColorType::L8 | ColorType::L16) {
        return Err(Error::InvalidParameter {
            name: "image",
            reason: format!(
                "{} must be a single-channel 8-bit or 16-bit grayscale image",
                path.display()
            ),
        });
    }
    let (width, height) = decoder.dimensions();
    Ok((height as usize, width as usize))
}

pub(crate) fn load_grayscale(
    path: impl AsRef<Path>,
    scaling: GrayscaleScaling,
) -> Result<Array2<f64>> {
    let path = path.as_ref();
    let image = image::open(path)?;
    match image {
        DynamicImage::ImageLuma8(buffer) => {
            let shape = (buffer.height() as usize, buffer.width() as usize);
            let scale = match scaling {
                GrayscaleScaling::NativeCounts => 1.0,
                GrayscaleScaling::Unit => 1.0 / u8::MAX as f64,
            };
            Array2::from_vec(
                shape,
                buffer
                    .into_raw()
                    .into_iter()
                    .map(|value| value as f64 * scale)
                    .collect(),
            )
        }
        DynamicImage::ImageLuma16(buffer) => {
            let shape = (buffer.height() as usize, buffer.width() as usize);
            let scale = match scaling {
                GrayscaleScaling::NativeCounts => 1.0,
                GrayscaleScaling::Unit => 1.0 / u16::MAX as f64,
            };
            Array2::from_vec(
                shape,
                buffer
                    .into_raw()
                    .into_iter()
                    .map(|value| value as f64 * scale)
                    .collect(),
            )
        }
        _ => Err(Error::InvalidParameter {
            name: "image",
            reason: format!(
                "{} must be a single-channel 8-bit or 16-bit grayscale image",
                path.display()
            ),
        }),
    }
}

pub(crate) fn load_grayscale_tiff_pages(
    path: impl AsRef<Path>,
    scaling: GrayscaleScaling,
) -> Result<Vec<Array2<f64>>> {
    let path = path.as_ref();
    let mut decoder = tiff::decoder::Decoder::new(BufReader::new(File::open(path)?))?;
    let mut pages = Vec::new();
    loop {
        let (width, height) = decoder.dimensions()?;
        if !matches!(decoder.colortype()?, tiff::ColorType::Gray(8 | 16)) {
            return Err(Error::InvalidParameter {
                name: "TIFF image",
                reason: format!(
                    "{} must contain only 8-bit or 16-bit grayscale pages",
                    path.display()
                ),
            });
        }
        let values = match decoder.read_image()? {
            tiff::decoder::DecodingResult::U8(values) => {
                let scale = match scaling {
                    GrayscaleScaling::NativeCounts => 1.0,
                    GrayscaleScaling::Unit => 1.0 / u8::MAX as f64,
                };
                values
                    .into_iter()
                    .map(|value| value as f64 * scale)
                    .collect()
            }
            tiff::decoder::DecodingResult::U16(values) => {
                let scale = match scaling {
                    GrayscaleScaling::NativeCounts => 1.0,
                    GrayscaleScaling::Unit => 1.0 / u16::MAX as f64,
                };
                values
                    .into_iter()
                    .map(|value| value as f64 * scale)
                    .collect()
            }
            _ => {
                return Err(Error::InvalidParameter {
                    name: "TIFF image",
                    reason: format!(
                        "{} must contain only unsigned 8-bit or 16-bit pages",
                        path.display()
                    ),
                });
            }
        };
        pages.push(Array2::from_vec((height as usize, width as usize), values)?);
        if !decoder.more_images() {
            break;
        }
        decoder.next_image()?;
    }
    Ok(pages)
}

/// Reads TIFF page headers without decoding their pixel buffers.
pub(crate) fn grayscale_tiff_page_shapes(path: impl AsRef<Path>) -> Result<Vec<(usize, usize)>> {
    let path = path.as_ref();
    let mut decoder = tiff::decoder::Decoder::new(BufReader::new(File::open(path)?))?;
    let mut shapes = Vec::new();
    loop {
        let (width, height) = decoder.dimensions()?;
        validate_grayscale_tiff_color(path, decoder.colortype()?)?;
        shapes.push((height as usize, width as usize));
        if !decoder.more_images() {
            break;
        }
        decoder.next_image()?;
    }
    Ok(shapes)
}

/// Decodes one TIFF page while leaving all other page pixel buffers unread.
pub(crate) fn load_grayscale_tiff_page(
    path: impl AsRef<Path>,
    page_index: usize,
    scaling: GrayscaleScaling,
) -> Result<Array2<f64>> {
    let path = path.as_ref();
    let mut decoder = tiff::decoder::Decoder::new(BufReader::new(File::open(path)?))?;
    let mut current_page = 0;
    loop {
        let (width, height) = decoder.dimensions()?;
        validate_grayscale_tiff_color(path, decoder.colortype()?)?;
        if current_page == page_index {
            let values = decode_grayscale_tiff_values(path, decoder.read_image()?, scaling)?;
            return Array2::from_vec((height as usize, width as usize), values);
        }
        if !decoder.more_images() {
            return Err(Error::InvalidMeasurements(format!(
                "TIFF {} does not contain page {page_index}",
                path.display()
            )));
        }
        decoder.next_image()?;
        current_page += 1;
    }
}

fn validate_grayscale_tiff_color(path: &Path, color_type: tiff::ColorType) -> Result<()> {
    if matches!(color_type, tiff::ColorType::Gray(8 | 16)) {
        Ok(())
    } else {
        Err(Error::InvalidParameter {
            name: "TIFF image",
            reason: format!(
                "{} must contain only 8-bit or 16-bit grayscale pages",
                path.display()
            ),
        })
    }
}

fn decode_grayscale_tiff_values(
    path: &Path,
    decoded: tiff::decoder::DecodingResult,
    scaling: GrayscaleScaling,
) -> Result<Vec<f64>> {
    match decoded {
        tiff::decoder::DecodingResult::U8(values) => {
            let scale = match scaling {
                GrayscaleScaling::NativeCounts => 1.0,
                GrayscaleScaling::Unit => 1.0 / u8::MAX as f64,
            };
            Ok(values
                .into_iter()
                .map(|value| value as f64 * scale)
                .collect())
        }
        tiff::decoder::DecodingResult::U16(values) => {
            let scale = match scaling {
                GrayscaleScaling::NativeCounts => 1.0,
                GrayscaleScaling::Unit => 1.0 / u16::MAX as f64,
            };
            Ok(values
                .into_iter()
                .map(|value| value as f64 * scale)
                .collect())
        }
        _ => Err(Error::InvalidParameter {
            name: "TIFF image",
            reason: format!(
                "{} must contain only unsigned 8-bit or 16-bit pages",
                path.display()
            ),
        }),
    }
}
