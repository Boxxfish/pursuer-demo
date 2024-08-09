use bevy::{
    prelude::*,
    render::{
        render_asset::RenderAssetUsages,
        render_resource::{Extent3d, TextureDimension, TextureFormat},
        texture::ImageSampler,
    },
};
use candle_core::{DType, Device, Tensor};

use crate::{
    agents::{update_observations, PlayerAgent, PursuerAgent},
    gridworld::{LevelLayout, GRID_CELL_SIZE},
    models::MeasureModel,
    net::{load_weights_into_net, NNWrapper},
    screens::GameScreen,
};

/// Plugin for Bayes filtering functionality.
pub struct FilterPlugin;

impl Plugin for FilterPlugin {
    fn build(&self, app: &mut App) {}
}

/// Adds playable functionality to `FilterPlugin`.
pub struct FilterPlayPlugin;

impl Plugin for FilterPlayPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, init_filter_net).add_systems(
            Update,
            (
                init_probs_viewer.run_if(resource_added::<LevelLayout>),
                update_filter_learned
                    .after(update_observations)
                    .run_if(resource_exists::<LevelLayout>),
                update_filter_manual
                    .after(update_observations)
                    .run_if(resource_exists::<LevelLayout>),
                update_probs_viewers,
                toggle_viewers,
            ),
        );
    }
}

/// Stores data for the filter.
#[derive(Component)]
pub struct BayesFilter {
    pub probs: Tensor,
    pub size: usize,
}

impl BayesFilter {
    pub fn new(size: usize) -> Self {
        let probs = (Tensor::ones(&[size, size], DType::F32, &Device::Cpu).unwrap()
            / (size * size) as f64)
            .unwrap();
        Self { probs, size }
    }

    pub fn reset(&mut self) {
        self.probs = (Tensor::ones(&[self.size, self.size], DType::F32, &Device::Cpu).unwrap()
            / (self.size * self.size) as f64)
            .unwrap();
    }
}

/// Causes the filter to use the learned model.
#[derive(Component)]
struct LearnedFilter;

/// Causes the filter to use manual rules.
#[derive(Component)]
struct ManualFilter;

/// Initializes the filter.
fn init_filter_net(mut commands: Commands) {
    commands.spawn((BayesFilter::new(16), ManualFilter));
}

fn apply_motion(probs: &Tensor, walls: &[bool], size: usize) -> candle_core::Result<Tensor> {
    let walls = Tensor::from_slice(
        &walls.iter().map(|b| *b as u8 as f32).collect::<Vec<_>>(),
        &[walls.len()],
        &Device::Cpu,
    )?
    .reshape(&[size, size])?;
    let kernel = Tensor::from_slice(
        &[0.25, 1., 0.25, 1., 1., 1., 0.25, 1., 0.25],
        &[1, 3, 3],
        &Device::Cpu,
    )?
    .reshape(&[1, 1, 3, 3])?
    .to_dtype(DType::F32)?;
    // Normalize by number of neighbors
    let denom = ((1. - walls)?
        .unsqueeze(0)?
        .unsqueeze(0)?
        .conv2d(&kernel, 1, 1, 1, 1)?
        .squeeze(0)?
        .squeeze(0)?
        + 0.001)?;
    let probs = (probs
        .unsqueeze(0)?
        .unsqueeze(0)?
        .conv2d(&kernel, 1, 1, 1, 1)?
        .squeeze(0)?
        .squeeze(0)?
        / denom)?;
    Ok(probs)
}

/// Updates filter probabilities for learned filters.
fn update_filter_learned(
    net_query: Query<&NNWrapper<MeasureModel>>,
    mut filter_query: Query<&mut BayesFilter, With<LearnedFilter>>,
    pursuer_query: Query<&PursuerAgent>,
    level: Res<LevelLayout>,
) {
    for mut filter in filter_query.iter_mut() {
        let model = net_query.single();
        if let Some(net) = &model.net {
            if let Ok(pursuer) = pursuer_query.get_single() {
                if pursuer.obs_timer.just_finished() {
                    if let Some((grid, objs, objs_attn)) = pursuer.observations.as_ref() {
                        // Apply motion model
                        let probs = apply_motion(&filter.probs, &level.walls, level.size).unwrap();

                        // Apply measurement model
                        let lkhd = net
                            .forward(
                                &grid.unsqueeze(0).unwrap(),
                                objs.as_ref().map(|t| t.unsqueeze(0).unwrap()).as_ref(),
                                objs_attn.as_ref().map(|t| t.unsqueeze(0).unwrap()).as_ref(),
                            )
                            .unwrap()
                            .squeeze(0)
                            .unwrap();
                        let probs = (probs * lkhd).unwrap();
                        let probs = (&probs
                            / probs.sum_all().unwrap().to_scalar::<f32>().unwrap() as f64)
                            .unwrap();

                        filter.probs = probs;
                    }
                }
            }
        }
    }
}

pub fn pos_to_grid(x: f32, y: f32, cell_size: f32) -> (usize, usize) {
    (
        (x / cell_size).round() as usize,
        (y / cell_size).round() as usize,
    )
}

/// Updates filter probabilities for manual filters.
fn update_filter_manual(
    mut filter_query: Query<&mut BayesFilter, With<ManualFilter>>,
    pursuer_query: Query<(&PursuerAgent, &GlobalTransform)>,
    player_query: Query<(Entity, &GlobalTransform), With<PlayerAgent>>,
    level: Res<LevelLayout>,
) {
    for mut filter in filter_query.iter_mut() {
        if let Ok((pursuer, pursuer_xform)) = pursuer_query.get_single() {
            if pursuer.obs_timer.just_finished() {
                if let Some(agent_state) = &pursuer.agent_state {
                    // Apply motion model
                    let probs = apply_motion(&filter.probs, &level.walls, level.size).unwrap();

                    // Apply measurement model
                    let size = level.size;
                    let mut lkhd = vec![vec![0.; level.size]; level.size];
                    for y in 0..size {
                        for x in 0..size {
                            let grid_lkhd = (!level.walls[y * size + x]) as u8 as f32;
                            let mut agent_lkhd = 1.;
                            let mut noise_lkhd = 1.;
                            let mut vis_lkhd = 1.;

                            if let Ok((player_e, player_xform)) = player_query.get_single() {
                                if agent_state.observing.contains(&player_e) {
                                    let player_pos = player_xform.translation().xy();
                                    let player_pos =
                                        pos_to_grid(player_pos.x, player_pos.y, GRID_CELL_SIZE);
                                    if player_pos != (x, y) {
                                        agent_lkhd = 0.;
                                    }
                                } else {
                                    // Cells within vision have 0% chance of agent being there
                                    agent_lkhd = 1. - agent_state.visible_cells[y * size + x];
                                    // All other cells are equally probable
                                    agent_lkhd /= size.pow(2) as f32
                                        - agent_state.visible_cells.iter().sum::<f32>();

                                    // If any noise sources are triggered, make the likelihood a normal distribution centered on it
                                    let pos = Vec2::new(x as f32, y as f32) * GRID_CELL_SIZE;
                                    let agent_pos = pursuer_xform.translation().xy();
                                    for obj_id in &agent_state.listening {
                                        let noise_obj =
                                            agent_state.noise_sources.get(obj_id).unwrap();
                                        let mean = noise_obj.pos;
                                        let var = (noise_obj.pos - agent_pos).length_squared();
                                        let diff = pos - mean;
                                        let exp = -((diff * diff) / (2. * var));
                                        let val = Vec2::new(exp.x.exp(), exp.y.exp())
                                            / f32::sqrt(2. * std::f32::consts::PI * var);
                                        noise_lkhd *= val.x * val.y;
                                    }

                                    // If any visual markers are moved, we can localize the player based on its start
                                    // position, end position, and how long it's been since the pursuer last looked at
                                    // it
                                    let max_speed = GRID_CELL_SIZE;
                                    for obj_id in &agent_state.observing {
                                        if agent_state.vm_data.contains_key(obj_id) {
                                            let vm_data = agent_state.vm_data[obj_id];
                                            let obs_obj = agent_state.objects[obj_id];
                                            if vm_data.last_seen_elapsed > 1.
                                                && !vm_data.pushed_by_self
                                            {
                                                let last_pos = vm_data.last_pos;
                                                let curr_pos = obs_obj.pos;
                                                let moved_amount =
                                                    (last_pos - curr_pos).length_squared();
                                                if moved_amount > 0.1 {
                                                    let curr_dist =
                                                        (pos - curr_pos).length_squared();
                                                    let max_dist = (max_speed
                                                        * vm_data.last_seen_elapsed)
                                                        .powi(2);
                                                    if curr_dist > max_dist {
                                                        vis_lkhd = 0.;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            lkhd[y][x] = grid_lkhd * agent_lkhd * noise_lkhd * vis_lkhd;
                        }
                    }
                    let lkhd = Tensor::from_slice(
                        &lkhd.into_iter().flatten().collect::<Vec<f32>>(),
                        &[size, size],
                        &Device::Cpu,
                    )
                    .unwrap();
                    let probs = (&probs * &lkhd).unwrap();
                    let probs = (&probs
                        / probs.sum_all().unwrap().to_scalar::<f32>().unwrap() as f64)
                        .unwrap();

                    filter.probs = probs;
                }
            }
        }
    }
}

/// Indicates visuals being used to show filter probabilities.
#[derive(Component)]
struct ProbsViewer {
    pub filter_e: Entity,
}

/// Initializes probs viewers.
fn init_probs_viewer(
    mut meshes: ResMut<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut commands: Commands,
    filter_query: Query<(Entity, &BayesFilter)>,
    level: Res<LevelLayout>,
) {
    for (e, filter) in filter_query.iter() {
        let size = filter.probs.shape().dims()[0];
        let mut img = Image::new(
            Extent3d {
                width: size as u32,
                height: size as u32,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            (0..(size * size))
                .flat_map(|_| [128, 128, 128, 255])
                .collect(),
            TextureFormat::Rgba8Unorm,
            RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
        );
        img.sampler = ImageSampler::nearest();
        let img = images.add(img);
        commands.spawn((
            GameScreen,
            ProbsViewer { filter_e: e },
            PbrBundle {
                mesh: meshes.add(Rectangle::new(
                    GRID_CELL_SIZE * level.size as f32,
                    GRID_CELL_SIZE * level.size as f32,
                )),
                material: materials.add(StandardMaterial {
                    base_color_texture: Some(img),
                    base_color: Color::WHITE.with_a(0.8),
                    unlit: true,
                    alpha_mode: AlphaMode::Blend,
                    cull_mode: None,
                    ..default()
                }),
                transform: Transform::default()
                    .with_translation(Vec3::new(
                        GRID_CELL_SIZE * (level.size - 1) as f32 / 2.,
                        GRID_CELL_SIZE * (level.size - 1) as f32 / 2.,
                        1.5,
                    ))
                    .with_rotation(Quat::from_rotation_x(std::f32::consts::PI)),
                visibility: Visibility::Hidden,
                ..default()
            },
        ));
    }
}

/// Updates probs viewers.
fn update_probs_viewers(
    filter_query: Query<&BayesFilter>,
    viewer_query: Query<(&ProbsViewer, &Handle<StandardMaterial>)>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    for (viewer, material) in viewer_query.iter() {
        let filter = filter_query.get(viewer.filter_e).unwrap();
        let probs = filter
            .probs
            .to_vec2::<f32>()
            .unwrap()
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        let data = probs
            .iter()
            .flat_map(|v| [(*v * 255.) as u8, (*v * 255.) as u8, (*v * 255.) as u8, 255])
            .collect::<Vec<_>>();
        if let Some(material) = materials.get_mut(material) {
            if let Some(image) = images.get_mut(material.base_color_texture.as_ref().unwrap()) {
                image.data = data;
                material.base_color_texture = material.base_color_texture.clone();
            }
        }
    }
}

/// Toggles viewers.
fn toggle_viewers(
    inpt: Res<ButtonInput<KeyCode>>,
    mut viewer_query: Query<&mut Visibility, With<ProbsViewer>>,
) {
    if inpt.just_pressed(KeyCode::Space) {
        for mut vis in viewer_query.iter_mut() {
            *vis = match vis.as_ref() {
                Visibility::Inherited | Visibility::Visible => Visibility::Hidden,
                Visibility::Hidden => Visibility::Visible,
            }
        }
    }
}
