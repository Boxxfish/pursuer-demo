use std::time::Duration;

use bevy::prelude::*;

pub struct MenuButtonPlugin;

impl Plugin for MenuButtonPlugin {
    fn build(&self, app: &mut App) {
        app.add_event::<MenuButtonPressedEvent>().add_systems(
            Update,
            (
                construct_menu_btns,
                update_labels,
                handle_menu_btns_on_change,
                handle_menu_btns,
            ),
        );
    }
}

pub enum MenuButtonContent {
    Label(String),
    Image(Handle<Image>),
}

impl Default for MenuButtonContent {
    fn default() -> Self {
        Self::Label("".into())
    }
}

/// A standard menu button.
#[derive(Default, Component)]
pub struct MenuButton {
    /// The text or image on the button.
    ///
    /// For consistency, if you want to change the content on the button, modify this instead of the underlying child.
    pub content: MenuButtonContent,
    hover_start: Duration,
}

/// A bundle for creating `MenuButton`s.
#[derive(Bundle)]
pub struct MenuButtonBundle {
    pub menu_button: MenuButton,
    pub button: ButtonBundle,
}

impl Default for MenuButtonBundle {
    fn default() -> Self {
        Self {
            menu_button: MenuButton::default(),
            button: ButtonBundle {
                style: Style {
                    display: Display::Flex,
                    width: Val::Px(128.),
                    max_height: Val::Px(128.),
                    padding: UiRect::axes(Val::Auto, Val::Px(4.)),
                    border: UiRect::all(Val::Px(1.)),
                    flex_direction: FlexDirection::Column,
                    align_items: AlignItems::Center,
                    ..default()
                },
                background_color: Color::WHITE.with_a(0.).into(),
                ..default()
            },
        }
    }
}

impl MenuButtonBundle {
    pub fn from_label(label: &str) -> Self {
        Self {
            menu_button: MenuButton {
                content: MenuButtonContent::Label(label.into()),
                ..default()
            },
            ..default()
        }
    }

    pub fn from_image(image: Handle<Image>) -> Self {
        Self {
            menu_button: MenuButton {
                content: MenuButtonContent::Image(image),
                ..default()
            },
            ..default()
        }
    }
}

/// Finishes constructing `MenuButton`s.
fn construct_menu_btns(
    btn_query: Query<(Entity, &MenuButton), Added<MenuButton>>,
    mut commands: Commands,
    asset_server: Res<AssetServer>,
) {
    let font = asset_server.load("fonts/montserrat/Montserrat-Regular.ttf");
    for (e, btn) in btn_query.iter() {
        commands.entity(e).with_children(|p| match &btn.content {
            MenuButtonContent::Label(label) => {
                p.spawn(TextBundle::from_section(
                    label.clone(),
                    TextStyle {
                        font: font.clone(),
                        font_size: 22.,
                        color: Color::WHITE,
                    },
                ));
            }
            MenuButtonContent::Image(image) => {
                p.spawn(ImageBundle {
                    style: Style {
                        width: Val::Px(120.),
                        height: Val::Px(120.),
                        ..default()
                    },
                    image: UiImage::new(image.clone()),
                    z_index: ZIndex::Local(1),
                    ..default()
                });
            }
        });
    }
}

/// Updates button content when it changes.
pub fn update_labels(
    btn_query: Query<(Entity, &MenuButton, &Children), Changed<MenuButton>>,
    mut text_query: Query<&mut Text>,
    image_query: Query<Entity, With<Handle<Image>>>,
    mut commands: Commands,
) {
    for (e, btn, children) in btn_query.iter() {
        for child in children.iter() {
            match &btn.content {
                MenuButtonContent::Label(label) => {
                    if let Ok(mut text) = text_query.get_mut(*child) {
                        text.sections[0].value = label.clone();
                    }
                }
                MenuButtonContent::Image(image) => {
                    if image_query.contains(*child) {
                        commands.entity(*child).insert(image.clone());
                    }
                }
            }
        }
    }
}

/// Sent by buttons when they're pressed.
#[derive(Event)]
pub struct MenuButtonPressedEvent {
    pub sender: Entity,
}

/// Handles menu button states on change.
fn handle_menu_btns_on_change(
    mut btn_query: Query<
        (
            Entity,
            &mut MenuButton,
            &Interaction,
            &mut BackgroundColor,
            &mut BorderColor,
        ),
        (Changed<Interaction>, With<Button>),
    >,
    mut ev_btn_pressed: EventWriter<MenuButtonPressedEvent>,
    time: Res<Time>,
) {
    for (e, mut btn, interaction, mut bg_color, mut border_color) in btn_query.iter_mut() {
        match interaction {
            Interaction::Pressed => {
                bg_color.0 = Color::WHITE.with_a(0.);
                border_color.0 = Color::WHITE;
                ev_btn_pressed.send(MenuButtonPressedEvent { sender: e });
            }
            Interaction::Hovered => {
                btn.hover_start = time.elapsed();
                bg_color.0 = Color::WHITE.with_a(0.);
                border_color.0 = Color::WHITE;
            }
            Interaction::None => {
                bg_color.0 = Color::WHITE.with_a(0.);
                border_color.0 = Color::WHITE.with_a(0.);
            }
        }
    }
}

/// Handles menu button states.
fn handle_menu_btns(
    mut btn_query: Query<(&mut MenuButton, &Interaction, &mut BackgroundColor), With<Button>>,
    time: Res<Time>,
) {
    for (btn, interaction, mut bg_color) in btn_query.iter_mut() {
        if interaction == &Interaction::Hovered {
            let elapsed = (time.elapsed() - btn.hover_start).as_secs_f32();
            let amount = (f32::cos(elapsed * std::f32::consts::TAU) + 1.) / 2.;
            bg_color.0 = Color::WHITE.with_a(0.05 * amount);
        }
    }
}
