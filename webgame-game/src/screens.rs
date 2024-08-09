use bevy::{asset::RecursiveDependencyLoadState, prelude::*};

use crate::{
    gridworld::{GameEndEvent, LevelLayout, LevelLoader, ShouldRun, GRID_CELL_SIZE},
    models::{MeasureModel, PolicyNet},
    net::NNWrapper,
    ui::{
        input_prompt::{InputPrompt, InputPromptBundle, InputType},
        menu_button::{MenuButtonBundle, MenuButtonPressedEvent},
        screen_transition::{FadeFinishedEvent, ScreenTransitionBundle, StartFadeEvent},
    },
};

/// Describes and handles logic for various screens.
pub struct ScreensPlayPlugin;

impl Plugin for ScreensPlayPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, init_ui)
            .add_systems(OnEnter(ScreenState::Loading), init_loading)
            .add_systems(OnExit(ScreenState::Loading), destroy_loading)
            .add_systems(
                Update,
                check_assets_loaded.run_if(in_state(ScreenState::Loading)),
            )
            .add_systems(OnEnter(ScreenState::TitleScreen), init_title_screen)
            .add_systems(OnExit(ScreenState::TitleScreen), destroy_title_screen)
            .add_systems(
                Update,
                (handle_title_screen_transition, handle_title_screen_btns)
                    .run_if(in_state(ScreenState::TitleScreen)),
            )
            .add_systems(OnEnter(ScreenState::LevelSelect), init_level_select)
            .add_systems(OnExit(ScreenState::LevelSelect), destroy_level_select)
            .add_systems(
                Update,
                (handle_level_select_transition, handle_level_select_btns)
                    .run_if(in_state(ScreenState::LevelSelect)),
            )
            .add_systems(OnEnter(ScreenState::Game), init_game)
            .add_systems(OnExit(ScreenState::Game), destroy_game)
            .add_systems(
                Update,
                (handle_game_transition, handle_game_btns, handle_game_end)
                    .run_if(in_state(ScreenState::Game)),
            )
            .add_systems(OnEnter(ScreenState::About), init_about)
            .add_systems(OnExit(ScreenState::About), destroy_about)
            .add_systems(
                Update,
                (handle_about_transition, handle_about_btns).run_if(in_state(ScreenState::About)),
            );
    }
}

/// The screens we can be on.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq, States, Default)]
pub enum ScreenState {
    #[default]
    Loading,
    TitleScreen,
    LevelSelect,
    Game,
    About,
}

const FONT_REGULAR: &str = "fonts/montserrat/Montserrat-Regular.ttf";
const FONT_BOLD: &str = "fonts/montserrat/Montserrat-Bold.ttf";

/// Denotes the loading screen.
#[derive(Component)]
struct LoadingScreen;

enum AssetType {
    Scene,
    Animation,
    Image,
}

/// A list of assets to load before the game runs.
const ASSETS_TO_LOAD: &[(&str, AssetType)] = &[
    (
        "characters/cyborgFemaleA.glb#Animation0",
        AssetType::Animation,
    ),
    (
        "characters/cyborgFemaleA.glb#Animation1",
        AssetType::Animation,
    ),
    ("characters/cyborgFemaleA.glb#Scene0", AssetType::Scene),
    ("characters/skaterMaleA.glb#Scene0", AssetType::Scene),
    ("furniture/wall.glb#Scene0", AssetType::Scene),
    ("furniture/wallDoorway.glb#Scene0", AssetType::Scene),
    ("furniture/doorway.glb#Scene0", AssetType::Scene),
    ("furniture/floorFull.glb#Scene0", AssetType::Scene),
    ("furniture/pottedPlant.glb#Scene0", AssetType::Scene),
    ("key.glb#Scene0", AssetType::Scene),
    (
        "arrow.png",
        AssetType::Image,
    ),
    (
        "input_prompts/keyboard_mouse/keyboard_wasd_outline.png",
        AssetType::Image,
    ),
    (
        "input_prompts/keyboard_mouse/keyboard_space_outline.png",
        AssetType::Image,
    ),
];

/// Handles toa ssets that must be loaded before the game runs.
#[derive(Resource)]
struct LoadingAssets {
    pub handles: Vec<UntypedHandle>,
}

fn init_loading(mut commands: Commands, asset_server: Res<AssetServer>) {
    let font = asset_server.load(FONT_REGULAR);
    commands
        .spawn((
            LoadingScreen,
            NodeBundle {
                style: Style {
                    width: Val::Percent(100.),
                    height: Val::Percent(100.),
                    display: Display::Flex,
                    flex_direction: FlexDirection::Column,
                    align_items: AlignItems::End,
                    justify_content: JustifyContent::End,
                    padding: UiRect::all(Val::Px(16.)),
                    ..default()
                },
                background_color: Color::BLACK.into(),
                ..default()
            },
        ))
        .with_children(|p| {
            p.spawn(TextBundle::from_section(
                "Loading...",
                TextStyle {
                    font: font.clone(),
                    font_size: 22.,
                    color: Color::WHITE,
                },
            ));
        });

    commands.spawn(NNWrapper::<PolicyNet>::with_sftensors(
        asset_server.load("p_net.safetensors"),
    ));
    let mut handles = Vec::new();
    for (path, asset_type) in ASSETS_TO_LOAD {
        match asset_type {
            AssetType::Scene => handles.push(asset_server.load::<Scene>(*path).untyped()),
            AssetType::Animation => {
                handles.push(asset_server.load::<AnimationClip>(*path).untyped())
            }
            AssetType::Image => handles.push(asset_server.load::<Image>(*path).untyped()),
        }
    }
    commands.insert_resource(LoadingAssets { handles });
}

/// Checks if assets are loaded, and transition to title screen if so.
fn check_assets_loaded(
    loading_assets: Option<Res<LoadingAssets>>,
    asset_server: Res<AssetServer>,
    mut commands: Commands,
    mut next_state: ResMut<NextState<ScreenState>>,
) {
    if let Some(loading_assets) = loading_assets {
        for handle in &loading_assets.handles {
            let load_state = asset_server.recursive_dependency_load_state(handle);
            match load_state {
                RecursiveDependencyLoadState::Loading | RecursiveDependencyLoadState::NotLoaded => {
                    return
                }
                _ => (),
            }
        }
        commands.remove_resource::<LoadingAssets>();
        next_state.0 = Some(ScreenState::TitleScreen);
    }
}

fn destroy_loading(mut commands: Commands, screen_query: Query<Entity, With<LoadingScreen>>) {
    commands.entity(screen_query.single()).despawn_recursive();
}

/// Denotes the title screen.
#[derive(Component)]
struct TitleScreen;

/// Actions that can be performed on the title screen.
#[derive(Component, Copy, Clone)]
enum TitleScreenAction {
    Start,
    About,
}

/// Holds the state to transition to when the transition finishes.
#[derive(Resource)]
struct TransitionNextState<T>(pub T);

/// Initializes UI elements that persist across scenes.
fn init_ui(mut commands: Commands) {
    commands.spawn(ScreenTransitionBundle::default());
    let cam_angle = (20.0_f32).to_radians();
    let cam_dist = 1200.;
    let size = 16;
    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_translation(Vec3::new(
                GRID_CELL_SIZE * (((size + 1) / 2) as f32),
                -cam_angle.sin() * cam_dist + GRID_CELL_SIZE * (((size + 1) / 2) as f32),
                cam_angle.cos() * cam_dist,
            ))
            .with_rotation(Quat::from_rotation_x(cam_angle)),
            projection: Projection::Perspective(PerspectiveProjection {
                fov: 0.4,
                ..default()
            }),
            ..default()
        },
        IsDefaultUiCamera,
    ));
}

fn init_title_screen(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut ev_start_fade: EventWriter<StartFadeEvent>,
) {
    let font_bold = asset_server.load(FONT_BOLD);
    ev_start_fade.send(StartFadeEvent { fade_in: true });

    commands
        .spawn((
            TitleScreen,
            NodeBundle {
                style: Style {
                    width: Val::Percent(100.),
                    height: Val::Percent(100.),
                    ..default()
                },
                background_color: Color::BLACK.into(),
                ..default()
            },
        ))
        .with_children(|p| {
            // Background
            p.spawn(ImageBundle {
                style: Style {
                    position_type: PositionType::Absolute,
                    margin: UiRect::all(Val::Auto),
                    ..default()
                },
                image: asset_server.load("ui/title_screen/background.png").into(),
                z_index: ZIndex::Local(0),
                ..default()
            });

            // Text elements
            p.spawn(NodeBundle {
                style: Style {
                    width: Val::Percent(100.),
                    height: Val::Percent(100.),
                    flex_direction: FlexDirection::Column,
                    display: Display::Flex,
                    ..default()
                },
                background_color: Color::BLACK.with_a(0.9).into(),
                z_index: ZIndex::Local(1),
                ..default()
            })
            .with_children(|p| {
                // Top text
                p.spawn(NodeBundle {
                    style: Style {
                        width: Val::Percent(100.),
                        height: Val::Percent(50.),
                        display: Display::Flex,
                        flex_direction: FlexDirection::Column,
                        align_items: AlignItems::Center,
                        justify_content: JustifyContent::End,
                        ..default()
                    },
                    ..default()
                })
                .with_children(|p| {
                    p.spawn(TextBundle::from_section(
                        "DEMO:",
                        TextStyle {
                            font: font_bold.clone(),
                            font_size: 40.,
                            color: Color::WHITE,
                        },
                    ));
                    p.spawn(TextBundle::from_section(
                        "PURSUER",
                        TextStyle {
                            font: font_bold.clone(),
                            font_size: 64.,
                            color: Color::WHITE,
                        },
                    ));
                });
                // Options
                p.spawn(NodeBundle {
                    style: Style {
                        width: Val::Px(128.),
                        height: Val::Percent(50.),
                        margin: UiRect::horizontal(Val::Auto),
                        display: Display::Flex,
                        flex_direction: FlexDirection::Column,
                        align_items: AlignItems::Center,
                        justify_content: JustifyContent::Start,
                        padding: UiRect::vertical(Val::Px(16.)),
                        ..default()
                    },
                    ..default()
                })
                .with_children(|p| {
                    for (action, label) in [
                        (TitleScreenAction::Start, "START"),
                        (TitleScreenAction::About, "ABOUT"),
                    ] {
                        p.spawn((action, MenuButtonBundle::from_label(label)));
                    }
                });
            });
        });
}

fn destroy_title_screen(mut commands: Commands, screen_query: Query<Entity, With<TitleScreen>>) {
    commands.entity(screen_query.single()).despawn_recursive();
}

fn handle_title_screen_btns(
    mut ev_btn_pressed: EventReader<MenuButtonPressedEvent>,
    action_query: Query<&TitleScreenAction>,
    mut ev_start_fade: EventWriter<StartFadeEvent>,
    mut commands: Commands,
) {
    for ev in ev_btn_pressed.read() {
        if let Ok(action) = action_query.get(ev.sender) {
            ev_start_fade.send(StartFadeEvent { fade_in: false });
            commands.insert_resource(TransitionNextState(*action));
        }
    }
}

fn handle_title_screen_transition(
    mut ev_fade_finished: EventReader<FadeFinishedEvent<ScreenState>>,
    mut commands: Commands,
    transition_state: Option<Res<TransitionNextState<TitleScreenAction>>>,
    mut next_state: ResMut<NextState<ScreenState>>,
) {
    for ev in ev_fade_finished.read() {
        if !ev.fade_in && ev.from_state == ScreenState::TitleScreen {
            commands.remove_resource::<TransitionNextState<TitleScreenAction>>();
            next_state.0 = match transition_state.as_ref().unwrap().0 {
                TitleScreenAction::Start => Some(ScreenState::LevelSelect),
                TitleScreenAction::About => Some(ScreenState::About),
            }
        }
    }
}
/// Denotes the game screen.
#[derive(Component)]
pub struct GameScreen;

enum GameAction {
    LevelSelect,
    Restart,
}

fn init_game(mut commands: Commands, mut ev_start_fade: EventWriter<StartFadeEvent>) {
    create_game_ui(&mut commands);

    ev_start_fade.send(StartFadeEvent { fade_in: true });
}

fn create_game_ui(commands: &mut Commands) {
    commands
        .spawn((
            GameScreen,
            NodeBundle {
                style: Style {
                    width: Val::Percent(100.),
                    height: Val::Percent(100.),
                    ..default()
                },
                ..default()
            },
        ))
        .with_children(|p| {
            p.spawn(NodeBundle {
                style: Style {
                    width: Val::Percent(100.),
                    height: Val::Percent(100.),
                    display: Display::Flex,
                    flex_direction: FlexDirection::Column,
                    padding: UiRect::all(Val::Px(8.)),
                    ..default()
                },
                ..default()
            })
            .with_children(|p| {
                p.spawn(NodeBundle {
                    style: Style {
                        width: Val::Percent(100.),
                        flex_direction: FlexDirection::Column,
                        align_items: AlignItems::Start,
                        ..default()
                    },
                    ..default()
                })
                .with_children(|p| {
                    for (label, input) in [
                        ("Move", InputType::WASD),
                        ("Toggle Filter", InputType::Space),
                    ] {
                        p.spawn(InputPromptBundle {
                            input_prompt: InputPrompt {
                                label: label.into(),
                                input,
                            },
                            ..default()
                        });
                    }
                });
                p.spawn(NodeBundle {
                    style: Style {
                        width: Val::Percent(100.),
                        height: Val::Percent(100.),
                        display: Display::Flex,
                        justify_content: JustifyContent::End,
                        align_items: AlignItems::Center,
                        flex_direction: FlexDirection::Column,
                        padding: UiRect::all(Val::Px(8.)),
                        ..default()
                    },
                    ..default()
                })
                .with_children(|p| {
                    p.spawn((MenuButtonBundle::from_label("BACK"),));
                });
            });
        });
}

fn destroy_game(mut commands: Commands, screen_query: Query<Entity, With<GameScreen>>) {
    for e in screen_query.iter() {
        commands.entity(e).despawn_recursive();
    }
    commands.remove_resource::<ShouldRun>();
    commands.remove_resource::<LevelLayout>();
}

fn handle_game_btns(
    mut ev_btn_pressed: EventReader<MenuButtonPressedEvent>,
    mut ev_start_fade: EventWriter<StartFadeEvent>,
    mut commands: Commands,
) {
    for _ in ev_btn_pressed.read() {
        commands.insert_resource(TransitionNextState(GameAction::LevelSelect));
        ev_start_fade.send(StartFadeEvent { fade_in: false });
    }
}

fn handle_game_end(
    mut ev_game_end: EventReader<GameEndEvent>,
    mut ev_start_fade: EventWriter<StartFadeEvent>,
    mut commands: Commands,
) {
    for ev in ev_game_end.read() {
        let state = if ev.player_won {
            GameAction::LevelSelect
        } else {
            GameAction::Restart
        };
        commands.insert_resource(TransitionNextState(state));
        ev_start_fade.send(StartFadeEvent { fade_in: false });
    }
}

fn handle_game_transition(
    transition_state: Option<Res<TransitionNextState<GameAction>>>,
    mut ev_fade_finished: EventReader<FadeFinishedEvent<ScreenState>>,
    mut next_state: ResMut<NextState<ScreenState>>,
    mut commands: Commands,
    level: Option<Res<LevelLayout>>,
    screen_query: Query<Entity, With<GameScreen>>,
    mut ev_start_fade: EventWriter<StartFadeEvent>,
) {
    for ev in ev_fade_finished.read() {
        if !ev.fade_in && ev.from_state == ScreenState::Game {
            commands.remove_resource::<TransitionNextState<GameAction>>();
            next_state.0 = match &transition_state.as_ref().unwrap().0 {
                GameAction::LevelSelect => Some(ScreenState::LevelSelect),
                GameAction::Restart => {
                    for e in screen_query.iter() {
                        commands.entity(e).despawn_recursive();
                    }
                    let level_: LevelLayout = level.as_ref().unwrap().as_ref().clone();
                    commands.remove_resource::<LevelLayout>();
                    commands.insert_resource(level_);
                    create_game_ui(&mut commands);
                    ev_start_fade.send(StartFadeEvent { fade_in: true });
                    Some(ScreenState::Game)
                }
            }
        }
    }
}

/// Denotes the level select screen.
#[derive(Component)]
struct LevelSelectScreen;

#[derive(Component, Clone)]
enum LevelSelectAction {
    Level(String),
    Back,
}

fn init_level_select(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut ev_start_fade: EventWriter<StartFadeEvent>,
) {
    ev_start_fade.send(StartFadeEvent { fade_in: true });

    commands
        .spawn((
            LevelSelectScreen,
            NodeBundle {
                style: Style {
                    width: Val::Percent(100.),
                    height: Val::Percent(100.),
                    ..default()
                },
                background_color: Color::BLACK.into(),
                ..default()
            },
        ))
        .with_children(|p| {
            // Background
            p.spawn(ImageBundle {
                style: Style {
                    position_type: PositionType::Absolute,
                    margin: UiRect::all(Val::Auto),
                    ..default()
                },
                image: asset_server.load("ui/title_screen/background.png").into(),
                z_index: ZIndex::Local(0),
                ..default()
            });

            // Main elements
            p.spawn(NodeBundle {
                style: Style {
                    width: Val::Percent(100.),
                    height: Val::Percent(100.),
                    flex_direction: FlexDirection::Column,
                    display: Display::Flex,
                    ..default()
                },
                background_color: Color::BLACK.with_a(0.95).into(),
                z_index: ZIndex::Local(1),
                ..default()
            })
            .with_children(|p| {
                p.spawn(NodeBundle {
                    style: Style {
                        display: Display::Flex,
                        flex_direction: FlexDirection::Row,
                        column_gap: Val::Px(16.),
                        row_gap: Val::Px(16.),
                        margin: UiRect::axes(Val::Auto, Val::Px(32.)),
                        ..default()
                    },
                    ..default()
                })
                .with_children(|p| {
                    for level in ["1", "2", "3", "4"] {
                        let image = asset_server.load(format!("ui/level_select/{level}.png"));
                        p.spawn((
                            MenuButtonBundle::from_image(image),
                            LevelSelectAction::Level(format!("levels/{level}.json")),
                        ));
                    }
                });

                p.spawn(NodeBundle {
                    style: Style {
                        width: Val::Px(128.),
                        margin: UiRect::axes(Val::Auto, Val::Auto),
                        ..default()
                    },
                    ..default()
                })
                .with_children(|p| {
                    p.spawn((
                        LevelSelectAction::Back,
                        MenuButtonBundle::from_label("BACK"),
                    ));
                });
            });
        });
}

fn destroy_level_select(
    mut commands: Commands,
    screen_query: Query<Entity, With<LevelSelectScreen>>,
) {
    commands.entity(screen_query.single()).despawn_recursive();
}

fn handle_level_select_btns(
    mut ev_btn_pressed: EventReader<MenuButtonPressedEvent>,
    action_query: Query<&LevelSelectAction>,
    mut ev_start_fade: EventWriter<StartFadeEvent>,
    mut commands: Commands,
) {
    for ev in ev_btn_pressed.read() {
        if let Ok(action) = action_query.get(ev.sender) {
            ev_start_fade.send(StartFadeEvent { fade_in: false });
            commands.insert_resource(TransitionNextState(action.clone()));
        }
    }
}

fn handle_level_select_transition(
    mut ev_fade_finished: EventReader<FadeFinishedEvent<ScreenState>>,
    mut commands: Commands,
    transition_state: Option<Res<TransitionNextState<LevelSelectAction>>>,
    mut next_state: ResMut<NextState<ScreenState>>,
) {
    for ev in ev_fade_finished.read() {
        if !ev.fade_in && ev.from_state == ScreenState::LevelSelect {
            commands.remove_resource::<TransitionNextState<LevelSelectAction>>();
            next_state.0 = match &transition_state.as_ref().unwrap().0 {
                LevelSelectAction::Level(level) => {
                    commands.insert_resource(LevelLoader::Path(level.clone()));
                    Some(ScreenState::Game)
                }
                LevelSelectAction::Back => Some(ScreenState::TitleScreen),
            }
        }
    }
}

/// Denotes the about screen.
#[derive(Component)]
struct AboutScreen;

fn init_about(
    mut commands: Commands,
    mut ev_start_fade: EventWriter<StartFadeEvent>,
    asset_server: Res<AssetServer>,
) {
    ev_start_fade.send(StartFadeEvent { fade_in: true });

    commands
        .spawn((
            AboutScreen,
            NodeBundle {
                style: Style {
                    width: Val::Percent(100.),
                    height: Val::Percent(100.),
                    ..default()
                },
                background_color: Color::BLACK.into(),
                ..default()
            },
        ))
        .with_children(|p| {
            // Background
            p.spawn(ImageBundle {
                style: Style {
                    position_type: PositionType::Absolute,
                    margin: UiRect::all(Val::Auto),
                    ..default()
                },
                image: asset_server.load("ui/title_screen/background.png").into(),
                z_index: ZIndex::Local(0),
                ..default()
            });

            // Text elements
            p.spawn(NodeBundle {
                style: Style {
                    width: Val::Percent(100.),
                    height: Val::Percent(100.),
                    flex_direction: FlexDirection::Column,
                    display: Display::Flex,
                    ..default()
                },
                background_color: Color::BLACK.with_a(0.95).into(),
                z_index: ZIndex::Local(1),
                ..default()
            })
            .with_children(|p| {
                p.spawn(NodeBundle {
                    style: Style {
                        flex_direction: FlexDirection::Column,
                        display: Display::Flex,
                        padding: UiRect::axes(Val::Px(32.), Val::Px(16.)),
                        ..default()
                    },
                    z_index: ZIndex::Local(1),
                    ..default()
                }).with_children(|p| {
                let font_regular = asset_server.load(FONT_REGULAR);
                let color = Color::WHITE;
                let heading = TextStyle {
                    font: font_regular.clone(),
                    font_size: 28.,
                    color,
                };
                let regular = TextStyle {
                    font: font_regular.clone(),
                    font_size: 16.,
                    color,
                };
                p.spawn(TextBundle::from_sections([
                    TextSection::new("About\n", heading.clone()),
                    TextSection::new("\nThis game demonstrates how machine learning can be used to create intelligent pursuer-type enemies.\n\n", regular.clone()),
                    TextSection::new(
                        "The pursuer is equipped with a discrete Bayes filter, allowing it to use evidence from its environment to track you down. It does this by generating a map of where it thinks you are and continuously updating it based on data from its sensors. You can toggle this map in-game to see where the pursuer thinks you are.\n\n", 
                        regular.clone()
                    ),
                    TextSection::new(
                        "The pursuer has also been trained to chase after you, using reinforcement learning.\n\n",
                        regular.clone()
                    ),
                    TextSection::new(
                        "Instructions\n",
                        heading.clone()
                    ),
                    TextSection::new(
                        "\nFind the key in each level and use it to escape. Be careful, though — there's someone locked in with you roaming the halls, and you REALLY don't want to come face to face with them!\n\n",
                        regular.clone()
                    ),
                ]));
                p.spawn(NodeBundle {
                    style: Style {
                        width: Val::Px(128.),
                        margin: UiRect::axes(Val::Auto, Val::Auto),
                        ..default()
                    },
                    ..default()
                }).with_children(|p| {
                    p.spawn(MenuButtonBundle::from_label("BACK"));
                });
            });
        });
    });
}

fn destroy_about(mut commands: Commands, screen_query: Query<Entity, With<AboutScreen>>) {
    commands.entity(screen_query.single()).despawn_recursive();
}

fn handle_about_btns(
    mut ev_btn_pressed: EventReader<MenuButtonPressedEvent>,
    mut ev_start_fade: EventWriter<StartFadeEvent>,
) {
    for _ in ev_btn_pressed.read() {
        ev_start_fade.send(StartFadeEvent { fade_in: false });
    }
}

fn handle_about_transition(
    mut ev_fade_finished: EventReader<FadeFinishedEvent<ScreenState>>,
    mut next_state: ResMut<NextState<ScreenState>>,
) {
    for ev in ev_fade_finished.read() {
        if !ev.fade_in && ev.from_state == ScreenState::About {
            next_state.0 = Some(ScreenState::TitleScreen);
        }
    }
}
