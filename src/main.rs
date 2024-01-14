//use image::io::Reader as ImageReader;
use clap::Parser;
use image::error::ImageError;
use image::imageops::FilterType;
use image::GenericImageView;
use image::RgbImage;
use indicatif::ProgressBar; //Iterator;
use rand::prelude::*;
use rayon::prelude::*;

#[derive(Parser, Debug)]
struct Args {
    #[arg(short, long)]
    src_path: String,

    #[arg(short, long)]
    dst_path: String,

    #[arg(short, long)]
    target_path: String,

    #[arg(short, long, default_value_t = 100)]
    num_iters: u32,

    #[arg(short = 'r', long, default_value_t = 1.0)]
    step_factor: f32,

    #[arg(short, long, default_value_t = 16)]
    batch_size: u32,

    #[arg(short, long, default_value_t)]
    palette: bool,

    #[arg(short, long, default_value_t)]
    verbose: bool,
}

fn project_colors(image: &RgbImage, dir: &[f32; 3]) -> Vec<f32> {
    image
        .pixels()
        .map(|p| p.0[0] as f32 * dir[0] + p.0[1] as f32 * dir[1] + p.0[2] as f32 * dir[2])
        .collect()
}

fn main() -> Result<(), ImageError> {
    let args = Args::parse();

    let src =
        image::open(&args.src_path).expect(&format!("Could not load image {}", &args.src_path));
    let tgt = image::open(&args.target_path)
        .expect(&format!("Could not load image {}", &args.target_path));

    let verbose = args.verbose;

    if verbose {
        println!("source image: {}x{}", src.width(), src.height());
        println!("target image: {}x{}", tgt.width(), tgt.height());
    }

    let mut src_buf = src.into_rgb8();

    let tgt_buf = if src_buf.dimensions() != tgt.dimensions() {
        if verbose {
            println!("resizing target...");
        }
        tgt.resize_exact(src_buf.width(), src_buf.height(), if args.palette { FilterType::Nearest } else { FilterType::Triangle })
            .into_rgb8()
    } else {
        tgt.into_rgb8()
    };

    let mut rng = rand::thread_rng();
    let normal_distr = rand_distr::StandardNormal;

    let num_iters = args.num_iters;
    let batch_size = args.batch_size;
    let mut step_factor = args.step_factor.clamp(0.01, 10.0);
    let num_pixels = (src_buf.width() * src_buf.height()) as usize;

    let mut advect_map = Vec::<[f32; 3]>::with_capacity(num_pixels);
    advect_map.resize(num_pixels, [0.0; 3]);

    let bar = ProgressBar::new(num_iters.into());
    let seq = (0..num_pixels).collect::<Vec<_>>();
    let mut maps = vec![Vec::<[f32; 3]>::with_capacity(num_pixels); batch_size as usize];
    maps.iter_mut()
        .for_each(|map| map.resize(num_pixels, [0.0; 3]));
    let mut sorted_src = Vec::with_capacity(num_pixels);
    let mut sorted_tgt = Vec::with_capacity(num_pixels);
    for _i in 0..num_iters {
        if !verbose {
            bar.inc(1);
        }

        advect_map.fill([0.0; 3]);
        // I tried parallelizing the batches, but the code is more convoluted and performance is
        // only better than simply parallelizing the sorts with large batch sizes
        for _j in 0..batch_size {
            // pick a direction
            let mut d = [
                rng.sample(normal_distr),
                rng.sample(normal_distr),
                rng.sample(normal_distr),
            ];
            let n = d.iter().map(|x| x * x).sum::<f32>().sqrt();
            d.iter_mut().for_each(|x| *x /= n);

            // project the colors and sort
            let src_projs = project_colors(&src_buf, &d);
            sorted_src.clone_from(&seq);
            sorted_src.par_sort_by(|&a, &b| src_projs[a].partial_cmp(&src_projs[b]).unwrap());

            let tgt_projs = project_colors(&tgt_buf, &d);
            sorted_tgt.clone_from(&seq);
            sorted_tgt.par_sort_by(|&a, &b| tgt_projs[a].partial_cmp(&tgt_projs[b]).unwrap());

            for (&src_idx, &tgt_idx) in sorted_src.iter().zip(sorted_tgt.iter()) {
                let a = step_factor * (tgt_projs[tgt_idx] - src_projs[src_idx]);
                for k in 0..3 {
                    advect_map[src_idx][k] += a * d[k];
                }
            }
        }

        let mut mean_v = [0.0; 3];
        for v in advect_map.iter_mut() {
            for k in 0..3 {
                v[k] /= batch_size as f32;
                mean_v[k] += v[k];
            }
        }

        mean_v.iter_mut().for_each(|c| *c /= num_pixels as f32);

        if verbose {
            println!(
                "iter {}: mean advection {:?} (step factor = {})",
                _i,
                mean_v.iter().map(|x| x * x).sum::<f32>().sqrt(),
                step_factor
            );
        }

        // apply the map
        src_buf
            .pixels_mut()
            .zip(&advect_map)
            .for_each(|(pixel, v)| {
                let p = &mut pixel.0;
                p[0] = (p[0] as i16 + v[0] as i16).clamp(0, u8::MAX.into()) as u8;
                p[1] = (p[1] as i16 + v[1] as i16).clamp(0, u8::MAX.into()) as u8;
                p[2] = (p[2] as i16 + v[2] as i16).clamp(0, u8::MAX.into()) as u8;
            });

        //src_buf.save(format!("{}/{}_{:03}.{}", std::path::Path::new(&args.dst_path).parent().unwrap().to_str().unwrap() , std::path::Path::new(&args.dst_path).file_stem().unwrap().to_str().unwrap(), _i, "jpg"))?;
    
        step_factor = 0.9 * step_factor + 0.1;
    }
    bar.finish();

    src_buf.save(&args.dst_path)?;

    Ok(())
}
