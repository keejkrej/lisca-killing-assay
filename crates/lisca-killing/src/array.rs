//! Full-frame fluorescence reduction. Same definition as transfection
//! `analysis.skipSegment`: area is the pixel count, background is the 10th
//! percentile, corrected fluorescence is the sum minus area times background.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaskedRoiStats {
    pub area: u32,
    pub intensity: f64,
    pub background: f64,
    pub corrected: f64,
}

pub const FULL_FRAME_BACKGROUND_QUANTILE: f64 = 0.1;

pub fn full_frame_roi_stats(frame: &[f64]) -> MaskedRoiStats {
    if frame.is_empty() {
        return MaskedRoiStats {
            area: 0,
            intensity: 0.0,
            background: 0.0,
            corrected: 0.0,
        };
    }
    let area = frame.len() as u32;
    let intensity: f64 = frame.iter().sum();
    let background = quantile(frame, FULL_FRAME_BACKGROUND_QUANTILE);
    MaskedRoiStats {
        area,
        intensity,
        background,
        corrected: intensity - f64::from(area) * background,
    }
}

fn quantile(values: &[f64], q: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    if values.len() == 1 {
        return values[0];
    }
    let Some(q) = q.is_finite().then_some(q.clamp(0.0, 1.0)) else {
        return 0.0;
    };
    let mut finite: Vec<f64> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if finite.is_empty() {
        return 0.0;
    }
    if finite.len() == 1 {
        return finite[0];
    }
    finite.sort_unstable_by(f64::total_cmp);
    let index = q * (finite.len() - 1) as f64;
    let lower_index = (index.floor() as usize).min(finite.len() - 1);
    let upper_index = (index.ceil() as usize).min(finite.len() - 1);
    let fraction = index - lower_index as f64;
    let lower = finite[lower_index];
    let upper = finite[upper_index];
    lower + fraction * (upper - lower)
}
