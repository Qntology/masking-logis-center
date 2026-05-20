use anyhow::Result;
use std::cmp::{max};

pub fn video_smart_resize(
    _t: u32, height: u32, width: u32,
    _temporal_patch_size: u32,
    factor: u32,
    min_pixels: u32,
    max_pixels: u32,
    _any: Option<u32>,
) -> Result<(u32, u32)> {
    let mut h_bar = (height as f64 / factor as f64).round() as u32 * factor;
    let mut w_bar = (width as f64 / factor as f64).round() as u32 * factor;

    h_bar = max(h_bar, factor);
    w_bar = max(w_bar, factor);

    let area = h_bar * w_bar;

    if area > max_pixels {
        let beta = (max_pixels as f64 / area as f64).sqrt();
        h_bar = ((h_bar as f64 * beta) / factor as f64).floor() as u32 * factor;
        w_bar = ((w_bar as f64 * beta) / factor as f64).floor() as u32 * factor;
    } else if area < min_pixels {
        let beta = (min_pixels as f64 / area as f64).sqrt();
        h_bar = ((h_bar as f64 * beta) / factor as f64).ceil() as u32 * factor;
        w_bar = ((w_bar as f64 * beta) / factor as f64).ceil() as u32 * factor;
    }

    Ok((max(h_bar, factor), max(w_bar, factor)))
}
