mod texture;
mod annotation;
mod input_controller;
mod global;
mod ui;
mod egui_tools;

use crate::egui_tools::EguiRenderer;
use egui_wgpu::{wgpu, ScreenDescriptor};
use std::sync::atomic::Ordering::Relaxed;

use std::sync::Arc;
use annotation::manager::LabelManager;
use geo::{coord, Line, Point};
use global::state::{EditState, EDIT_STATE};
use i_triangle::{i_overlay::{core::fill_rule::FillRule, i_float::{f32_point::F32Point, f64_point::F64Point}}, triangulation::float::FloatTriangulate};
use input_controller::Mouse::{get_shift_state, set_shift_down, set_shift_up, KeyState, MousePosition, MouseState};
use ui::ui_manager::{UIManager, UIElement, UIArea, UIButton, ScreenCoord};
// lib.rs
use winit::{event::{ElementState, KeyEvent, MouseButton, WindowEvent}, keyboard::{Key, KeyLocation}, platform::modifier_supplement::KeyEventExtModifierSupplement, window::Window};
use wgpu::{util::DeviceExt, MemoryHints};

// use crate::ui::


#[repr(C)]
#[derive(Default, Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Position {
    x : f32,
    y : f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Zoom(f32);

impl Default for Zoom {
    fn default() -> Self {
        return Zoom(1.0);
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AspectRatio(f32);
impl Default for AspectRatio {
    fn default() -> Self {
        return AspectRatio(1.0);
    }
}

#[repr(C)]
#[derive(Default, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Camera {
    pos : Position,
    zoom : Zoom,
    aspect_ratio : AspectRatio,
}



// lib.rs
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    tex_coords: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct LineVertex {
    position: [f32; 3],
}

impl LineVertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<LineVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    format: wgpu::VertexFormat::Float32x3,
                    shader_location: 0,
                }
            ],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct PointVertex {
    position: [f32; 3],
    value: [f32; 3]
}

impl PointVertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<PointVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ]
        }
    }
}

// lib.rs
impl Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                }
            ]
        }
    }
}


const VERTICES: &[Vertex] = &[
    Vertex { position: [-1.0, 1.0, 0.0], tex_coords: [0.0, 0.0]},
    Vertex { position: [-1.0, -1.0, 0.0], tex_coords: [0.0, 1.0]},
    Vertex { position: [1.0, -1.0, 0.0], tex_coords: [1.0, 1.0]},
    Vertex { position: [1.0, 1.0, 0.0], tex_coords: [1.0, 0.0]},
];

const INDICES: &[u16] = &[
    0, 1, 2,
    0, 2, 3,
    // 2, 3, 4,
];

#[derive(PartialEq, Eq)]
enum Selection {
    None,
    Selected(usize),
    MultiSelection(Vec<usize>),
}


pub struct State<'a> {
    surface: wgpu::Surface<'a>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,


    // EGUI
    egui_renderer : egui_tools::EguiRenderer,
    scale_factor : f32,
    // The window must be declared after the surface so
    // it gets dropped after it as the surface contains
    // unsafe references to the window's resources.
    // pub window: &'a Window,

    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,

    diffuse_bind_group: wgpu::BindGroup, // NEW!
    diffuse_texture: texture::Texture, // NEW


    prev_mouse_pos: MousePosition,
    prev_mouse_state : MouseState,
    prev_RMB_state : MouseState,
    prev_MMB_state : MouseState,

    camera: Camera,
    camera_bind_group: wgpu::BindGroup,
    camera_buffer : wgpu::Buffer,


    label_manager : LabelManager,
    label_selected : Selection,

    ui_manager : UIManager,
}

impl<'a> State<'a> {

    // Creating some of the wgpu types requires async code
    // ...
    pub async fn new(window: Arc<Window>) -> State<'a> {
        let size = window.inner_size();

        // The instance is a handle to our GPU
        // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = egui_wgpu::wgpu::Instance::new(wgpu::InstanceDescriptor {
            #[cfg(not(target_arch = "wasm32"))]
            backends: wgpu::Backends::PRIMARY,
            #[cfg(target_arch = "wasm32")]
            backends: wgpu::Backends::GL,
            ..Default::default()
        });

        let surface = instance.create_surface(Arc::clone(&window)).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::empty(),
                    // WebGL doesn't support all of wgpu's features, so if
                    // we're building for the web, we'll have to disable some.
                    required_limits: if cfg!(target_arch = "wasm32") {
                        wgpu::Limits::downlevel_webgl2_defaults()
                    } else {
                        wgpu::Limits::default()
                    },
                    memory_hints : MemoryHints::MemoryUsage ,
                    label: None,
                },
                None, // Trace path
            )
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        // Shader code in this tutorial assumes an sRGB surface texture. Using a different
        // one will result in all the colors coming out darker. If you want to support non
        // sRGB surfaces, you'll need to account for that when drawing to the frame.
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        // Loads a tree
        let diffuse_bytes = include_bytes!("../Textures/happytree.png"); // CHANGED!
        let diffuse_texture = texture::Texture::from_bytes(&device, &queue, diffuse_bytes, "happy-tree.png").unwrap(); // CHANGED!

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        // This should match the filterable field of the
                        // corresponding Texture entry above.
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });

        let diffuse_bind_group = device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                layout: &texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
                    }
                ],
                label: Some("diffuse_bind_group"),
            }
        );

        // Camera Buffer
        let mut camera_uniform = Camera::default();

        let camera_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Camera Buffer"),
                contents: bytemuck::cast_slice(&[camera_uniform]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }
        );

        let camera_bind_group_layout = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("camera_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }
                ] ,
            }
        );

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                }
            ],
            label: Some("camera_bind_group"),
        });

        // Shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let render_pipeline_layout = device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[
                    &texture_bind_group_layout,
                    &camera_bind_group_layout,
                ],
                push_constant_ranges: &[],
            }
        );

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main", // 1.
                buffers: &[
                    Vertex::desc(),
                ],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState { // 3.
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState { // 4.
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList, // 1.
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw, // 2.
                cull_mode: Some(wgpu::Face::Back),
                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: None, // 1.
            multisample: wgpu::MultisampleState {
                count: 1, // 2.
                mask: !0, // 3.
                alpha_to_coverage_enabled: false, // 4.
            },
            multiview: None, // 5.
            cache: None, // 6.
        });


        let label_manager = annotation::manager::setup_label_manager(&device, &config, &camera_bind_group_layout);
        let ui_manager = ui::ui_manager::setup_ui(&device, &config);

        let vertex_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(VERTICES),
                usage: wgpu::BufferUsages::VERTEX,
            }
        );

        let index_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: bytemuck::cast_slice(INDICES),
                usage: wgpu::BufferUsages::INDEX,
            }
        );



        // EGUI
        let egui_renderer = EguiRenderer::new(&device, config.format, None, 1, &window);

        let num_indices = INDICES.len() as u32;
        Self {
            surface,
            device,
            queue,
            config,
            size,
            egui_renderer,
            scale_factor : 1.0,

            render_pipeline,
            vertex_buffer,
            index_buffer,
            num_indices,
            diffuse_bind_group,
            diffuse_texture,

            prev_mouse_state : MouseState::Released,
            prev_RMB_state : MouseState::Released,
            prev_MMB_state : MouseState::Released,
            prev_mouse_pos : MousePosition {x : 0, y : 0},
            camera : camera_uniform,
            camera_buffer,
            camera_bind_group,

            label_manager,
            label_selected : Selection::None,

            ui_manager,
        }
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);

            global::state::WINDOW_SIZE_X.store(new_size.width, std::sync::atomic::Ordering::Relaxed);
            global::state::WINDOW_SIZE_Y.store(new_size.height, std::sync::atomic::Ordering::Relaxed);

            self.camera.aspect_ratio.0 = new_size.width as f32 / new_size.height as f32;
        }
    }

    pub fn input(&mut self, window : &Window, event: &WindowEvent) -> bool {
        if self.egui_renderer.handle_input(window, event) {
            return true;
        }

        match event {
            WindowEvent::MouseInput { device_id : _, state, button} => {
                match button {
                    MouseButton::Left => {
                        match state {
                            ElementState::Pressed => {input_controller::Mouse::set_mouse_down()},
                            ElementState::Released => {input_controller::Mouse::set_mouse_up()},
                        }
                    },
                    MouseButton::Right => {
                        match state {
                            ElementState::Pressed => {input_controller::Mouse::set_RMB_down()},
                            ElementState::Released => {input_controller::Mouse::set_RMB_up()},
                        }
                    },
                    MouseButton::Middle => {
                        match state {
                            ElementState::Pressed => {input_controller::Mouse::set_mmb_down()},
                            ElementState::Released => {input_controller::Mouse::set_mmb_up()},
                        }
                    }
                    _ => {},
                }
            },
            WindowEvent::ModifiersChanged(new_modifiers) => {
                match new_modifiers.lshift_state() {
                    winit::keyboard::ModifiersKeyState::Pressed => { set_shift_down(); },
                    _ => { set_shift_up(); },
                }
            }
            WindowEvent::PinchGesture { device_id, delta, phase } => {
                let change : f32 = 1.00 + (-delta / 5.0) as f32;
                self.camera.zoom.0 *= change;
            },
            WindowEvent::MouseWheel { device_id, delta, phase } => {
                match delta {
                    winit::event::MouseScrollDelta::LineDelta(_x, y) => {
                        let prev_camera_zoom = self.camera.zoom;
                        let change : f32 = 1.00 + (y / 100.0);
                        self.camera.zoom.0 *= change;

                        // Differences
                        let prev_viewed_width : f32 = self.size.width as f32 * prev_camera_zoom.0;
                        let new_viewed_width : f32 = self.size.width as f32 * self.camera.zoom.0;
                        let width_difference = new_viewed_width - prev_viewed_width;

                        let prev_viewed_height : f32 = self.size.height as f32 * prev_camera_zoom.0;
                        let new_viewed_height : f32 = self.size.height as f32 * self.camera.zoom.0;
                        let height_difference = new_viewed_height - prev_viewed_height;

                        // Zooms the camera on the mouse
                        let mouse_pos = input_controller::Mouse::get_mouse_pos();
                        let camera_x_ratio = -1.0 + ((mouse_pos.x * 2) as f32 / self.size.width as f32);
                        let camera_y_ratio = -1.0 + ((mouse_pos.y * 2) as f32 / self.size.height as f32);

                        self.camera.pos.x -= (width_difference * camera_x_ratio) / 2.0 * self.camera.aspect_ratio.0;
                        self.camera.pos.y += (height_difference * camera_y_ratio) / 2.0;
                    },
                    winit::event::MouseScrollDelta::PixelDelta(scroll_position) => {
                        self.camera.pos.x += (-scroll_position.x / 2.0) as f32;
                        self.camera.pos.y += (scroll_position.y / 2.0) as f32;
                    }
                }

                self.camera.zoom.0 = f32::min(self.camera.zoom.0, 10.0);
                self.camera.zoom.0 = f32::max(self.camera.zoom.0, 0.01);
            },
            WindowEvent::CursorLeft { device_id } => {
                self.label_manager.get_edit_label().remove_mouse_point();
            }
            WindowEvent::CursorMoved { device_id : _, position } => {
                let new_pos = input_controller::Mouse::MousePosition {
                    x : position.x as isize,
                    y : position.y as isize,
                };
                let (x, y) = self.window_to_world_pos(position.x as f32, -position.y as f32);
                self.label_manager.get_edit_label().add_mouse_point(x, y);
                input_controller::Mouse::set_mouse_pos(new_pos);
            },
            _ => (),
        }

        true
    }

    pub fn window_to_world_pos(&self, mut x : f32, mut y : f32) -> (f32, f32) {
        // Must be converted to image space
        x /= (self.size.width / 2) as f32;
        y /= (self.size.height / 2) as f32;

        // Transform to the middle of the screen
        x -= 1.0;
        y += 1.0;

        x *= self.camera.zoom.0;
        y *= self.camera.zoom.0;

        x *= self.camera.aspect_ratio.0;

        // Transform to camera position
        x += self.camera.pos.x / ((self.size.width / 2) as f32);
        y += self.camera.pos.y / ((self.size.height / 2) as f32);

        return (x, y);
    }


    pub fn update(&mut self) {
        let mouse_state = input_controller::Mouse::get_mouse_state();
        let mmb_state = input_controller::Mouse::get_MMB_state();
        let rmb_state = input_controller::Mouse::get_RMB_state();
        let mouse_pos = input_controller::Mouse::get_mouse_pos();

        let lmb_click : bool = mouse_state == MouseState::Pressed && self.prev_mouse_state == MouseState::Released;
        let rmb_click : bool = rmb_state == MouseState::Pressed && self.prev_RMB_state == MouseState::Released;

        let mut input_already_captured = false;

        if !input_already_captured && lmb_click {
            // Propagates the click and captures the input
            if self.ui_manager.propagate_click(mouse_pos.x as u32, mouse_pos.y as u32) {
                input_already_captured = true;
            }
        }

        if !input_already_captured && mmb_state == MouseState::Pressed {
            let change_in_pos = mouse_pos - self.prev_mouse_pos;

            self.camera.pos.x -= (change_in_pos.x as f32) * self.camera.zoom.0;
            self.camera.pos.y += (change_in_pos.y as f32) * self.camera.zoom.0;

            input_already_captured = true;
        }

        self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[self.camera]));

        let edit_state : EditState = EDIT_STATE.load(std::sync::atomic::Ordering::Relaxed);

        if !input_already_captured && lmb_click {
            // These points need to be translated to image space...
            let (x, y) = self.window_to_world_pos(mouse_pos.x as f32, -mouse_pos.y as f32);

            match edit_state {
                EditState::New | EditState::Additive | EditState::Subtract => {
                    let edit_label = self.label_manager.get_edit_label();
                    edit_label.add_point(x, y);
                    input_already_captured = true;
                },
                EditState::Select => {
                    let mouse_coord = coord!{x : x, y: y};

                    // Must hold shift for this..
                    if let Some(poly_index) = self.label_manager.get_poly_at_point(mouse_coord) {
                        // Ensures it is not already in the list
                        let not_in = match &self.label_selected {
                            Selection::Selected(selected) => { selected != &poly_index},
                            Selection::MultiSelection(selections) => { !selections.contains(&poly_index) },
                            _ => true,
                        } || get_shift_state() == KeyState::Released;

                        // Adds it to the list
                        if not_in {
                            self.label_selected = match &mut self.label_selected {
                                Selection::None => { Selection::Selected(poly_index) },
                                Selection::Selected(selected) => {
                                    match get_shift_state() {
                                        KeyState::Pressed => { Selection::MultiSelection([*selected, poly_index].to_vec()) },
                                        KeyState::Released => { Selection::Selected(poly_index) }
                                    }
                                },
                                Selection::MultiSelection(selections) => {
                                    match get_shift_state() {
                                        KeyState::Pressed => {
                                            selections.push(poly_index); // This should be fixed!
                                            Selection::MultiSelection(selections.clone())
                                        },
                                        KeyState::Released => { Selection::Selected(poly_index) }
                                    }
                                },
                            }
                        }
                    } else {
                        self.label_selected = Selection::None;
                    }
                },
            }

            input_already_captured = true;
        }

        if !input_already_captured && rmb_click {
            self.label_manager.propagate_edit_label(&self.label_selected, &edit_state);
            input_already_captured = true;
        }

        // Saving previous states
        self.prev_mouse_pos = mouse_pos;
        self.prev_RMB_state = rmb_state;
        self.prev_MMB_state = mmb_state;
        self.prev_mouse_state = mouse_state;
    }

    pub fn render(&mut self, window : &Window) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;

        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            // NEW!
            render_pass.set_pipeline(&self.render_pipeline); // 2.
            render_pass.set_bind_group(0, &self.diffuse_bind_group, &[]); // NEW!
            render_pass.set_bind_group(1, &self.camera_bind_group, &[]); // NEW!
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16); // 1.
            render_pass.draw_indexed(0..self.num_indices, 0, 0..1); // 2.

            self.label_manager.render(
                &self.device, &mut render_pass, &mut self.queue, &self.camera_bind_group, &self.camera, &self.label_selected,
            );
            self.ui_manager.render(&mut render_pass, &mut self.queue);
        }

        {
            self.egui_renderer.begin_frame(&window);

            egui::Window::new("winit + egui + wgpu says hello!")
                .resizable(true)
                .vscroll(true)
                .default_open(true)
                .show(self.egui_renderer.context(), |ui| {
                    ui.label("Label!");

                    if ui.button("Button!").clicked() {
                        println!("boom!")
                    }

                    ui.separator();
                    ui.horizontal(|ui| {
                        ui.label(format!(
                            "Pixels per point: {}",
                            self.egui_renderer.context().pixels_per_point()
                        ));
                        if ui.button("-").clicked() {
                            self.scale_factor = (self.scale_factor - 0.1).max(0.3);
                        }
                        if ui.button("+").clicked() {
                            self.scale_factor = (self.scale_factor + 0.1).min(3.0);
                        }
                    });

                    ui.separator();
                    ui.horizontal(|ui| {
                        if ui.button("Select").clicked() {
                            EDIT_STATE.store(EditState::Select, Relaxed);
                        };
                        if ui.button("New").clicked() {
                            EDIT_STATE.store(EditState::New, Relaxed);
                        };
                        if ui.button("Add").clicked() {
                            EDIT_STATE.store(EditState::Additive, Relaxed);
                        };
                        if ui.button("Subtract").clicked() {
                            EDIT_STATE.store(EditState::Subtract, Relaxed);
                        };
                    });
                    ui.separator();
                    if ui.button("Merge").clicked() {
                        // Merge
                        self.label_manager.merge_selected_labels(&self.label_selected);
                        self.label_selected = Selection::None;
                    };
                    if ui.button("Split").clicked() {
                        // Merge
                        self.label_manager.split_selected_labels(&self.label_selected);
                        self.label_selected = Selection::None;
                    }
                });

            let screen_descriptor = ScreenDescriptor {
                size_in_pixels: [self.config.width, self.config.height],
                pixels_per_point: window.scale_factor() as f32
                * self.scale_factor,
            };

            // let surface_texture = self
            //     .surface
            //     .get_current_texture()
            //     .expect("Failed to acquire next swap chain texture");

            let surface_view = output
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());

            self.egui_renderer.end_frame_and_draw(
                &self.device,
                &self.queue,
                &mut encoder,
                window,
                &surface_view,
                screen_descriptor,
            );
        }

        // submit will accept anything that implements IntoIter
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }


}
