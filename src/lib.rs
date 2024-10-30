mod texture;
mod annotation;
mod input_controller;
mod global;

use std::sync::Arc;
use annotation::manager::LabelManager;
use geo::{Line, Point};
use i_triangle::{i_overlay::{core::fill_rule::FillRule, i_float::{f32_point::F32Point, f64_point::F64Point}}, triangulation::float::FloatTriangulate};
use input_controller::Mouse::{MousePosition, MouseState};
// lib.rs
use winit::{event::{ElementState, MouseButton, WindowEvent}, window::Window};
use wgpu::{util::DeviceExt, MemoryHints};

#[repr(C)]
#[derive(Default, Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Position {
    x : f32,
    y : f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Zoom(f32);

impl Default for Zoom {
    fn default() -> Self {
        return Zoom(1.0);
    }
}

#[repr(C)]
#[derive(Default, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Camera {
    pos : Position,
    zoom : Zoom,
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

enum Selection {
    None,
    Selected(usize),
}


pub struct State<'a> {
    surface: wgpu::Surface<'a>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    // The window must be declared after the surface so
    // it gets dropped after it as the surface contains
    // unsafe references to the window's resources.
    // pub window: &'a Window,

    render_pipeline: wgpu::RenderPipeline,
    render_pipeline_2 : wgpu::RenderPipeline,
    render_pipeline_3 : wgpu::RenderPipeline,
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
}

impl<'a> State<'a> {
    // Creating some of the wgpu types requires async code
    // ...
    pub async fn new(window: Arc<Window>) -> State<'a> {
        let size = window.inner_size();

        // The instance is a handle to our GPU
        // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
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


        // Shader_2
        let shader_2 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Polygon Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("polygon_shader.wgsl").into()),
        });

        let render_pipeline_layout_2 = device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("Polygon Render Pipeline Layout"),
                bind_group_layouts: &[
                    &camera_bind_group_layout,
                ],
                push_constant_ranges: &[],
            }
        );

        let render_pipeline_2 = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline 2"),
            layout: Some(&render_pipeline_layout_2),
            vertex: wgpu::VertexState {
                module: &shader_2,
                entry_point: "vs_main", // 1.
                buffers: &[
                    PointVertex::desc(),
                ],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState { // 3.
                module: &shader_2,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState { // 4.
                    format: config.format,
                    // blend: Some(wgpu::BlendState::REPLACE),
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList, // 1.
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                // cull_mode: Some(wgpu::Face::),
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

        // Render Pipeline 3
        let shader_3 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Line Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("line_shader.wgsl").into()),
        });

        let render_pipeline_layout_3 = device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("Line Render Pipeline Layout"),
                bind_group_layouts: &[
                    &camera_bind_group_layout,
                ],
                push_constant_ranges: &[],
            }
        );

        let render_pipeline_3 = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline 3"),
            layout: Some(&render_pipeline_layout_3),
            vertex: wgpu::VertexState {
                module: &shader_3,
                entry_point: "vs_main", // 1.
                buffers: &[
                    LineVertex::desc(),
                ],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState { // 3.
                module: &shader_3,
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
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
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



        // new()
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


        let num_indices = INDICES.len() as u32;


        Self {
            surface,
            device,
            queue,
            config,
            size,

            render_pipeline,
            render_pipeline_2,
            render_pipeline_3,
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

            label_manager : LabelManager::new(),
            label_selected : Selection::None,
        }
    }


    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }


    pub fn input(&mut self, event: &WindowEvent) -> bool {
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
            WindowEvent::PinchGesture { device_id, delta, phase } => {
                let change : f32 = 1.00 + (-delta / 5.0) as f32;
                self.camera.zoom.0 *= change;
            },
            WindowEvent::MouseWheel { device_id, delta, phase } => {
                match delta {
                    winit::event::MouseScrollDelta::LineDelta(_x, y) => {
                        let change : f32 = 1.00 + (y / 2000.0);
                        self.camera.zoom.0 *= change;
                    },
                    winit::event::MouseScrollDelta::PixelDelta(scroll_position) => {
                        self.camera.pos.x += (-scroll_position.x / 2.0) as f32;
                        self.camera.pos.y += (scroll_position.y / 2.0) as f32;
                    }
                }

                self.camera.zoom.0 = f32::min(self.camera.zoom.0, 10.0);
                self.camera.zoom.0 = f32::max(self.camera.zoom.0, 0.01);
            },
            WindowEvent::CursorMoved { device_id : _, position } => {
                let new_pos = input_controller::Mouse::MousePosition {
                    x : position.x as isize,
                    y : position.y as isize,
                };
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


        if mmb_state == MouseState::Pressed {
            let change_in_pos = mouse_pos - self.prev_mouse_pos;

            self.camera.pos.x -= (change_in_pos.x as f32) * self.camera.zoom.0;
            self.camera.pos.y += (change_in_pos.y as f32) * self.camera.zoom.0;
        }

        self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[self.camera]));


        if self.prev_mouse_state== MouseState::Released && mouse_state == MouseState::Pressed {
            let label_index = match self.label_selected {
                Selection::None => {
                    self.label_manager.get_new_label()
                },
                Selection::Selected(index) => {
                    index
                }
            };
            self.label_selected = Selection::Selected(label_index);

            // These points need to be translated to image space...
            let (x, y) = self.window_to_world_pos(mouse_pos.x as f32, -mouse_pos.y as f32);

            let label = self.label_manager.get_label_at_index(label_index);
            label.point_list.add_point(x, y);
        }

        if self.prev_RMB_state == MouseState::Released && rmb_state == MouseState::Pressed {
            self.label_selected = Selection::None;
        }

        self.prev_mouse_pos = mouse_pos;
        self.prev_RMB_state = rmb_state;
        self.prev_MMB_state = mmb_state;
        self.prev_mouse_state = mouse_state;
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
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


            let labels = self.label_manager.get_labels();
            // Draw the Lines
            for label in labels {
                if let Some((line_points, line_indices)) = label.get_draw_line(self.camera.zoom.0) {
                    let vertex_buffer_3 = self.device.create_buffer_init(
                        &wgpu::util::BufferInitDescriptor {
                            label: Some("Vertex Buffer"),
                            contents: bytemuck::cast_slice(line_points.as_slice()),
                            usage: wgpu::BufferUsages::VERTEX,
                        }
                    );

                    let index_buffer_3 = self.device.create_buffer_init(
                        &wgpu::util::BufferInitDescriptor {
                            label: Some("Index Buffer"),
                            contents: bytemuck::cast_slice(line_indices.as_slice()),
                            usage: wgpu::BufferUsages::INDEX,
                        }
                    );
                    render_pass.set_pipeline(&self.render_pipeline_3);
                    render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
                    render_pass.set_vertex_buffer(0, vertex_buffer_3.slice(..));
                    render_pass.set_index_buffer(index_buffer_3.slice(..), wgpu::IndexFormat::Uint16);
                    render_pass.draw_indexed(0..line_indices.len() as u32, 0, 0..1); // 2.
                }
            }


            for label in labels {
                if let Some(tris) = label.to_point_vertices() {
                    let tri_vertices : &[PointVertex] = &tris[0..tris.len()];

                    // I_Triangle
                    let mut i_vec : Vec<F64Point> = Vec::new();

                    for tri in tri_vertices {
                        i_vec.push(
                            F64Point {
                                x : tri.position[0] as f64,
                                y : tri.position[1] as f64,
                            }
                        );
                    }

                    let shape = [
                        i_vec,
                        [].to_vec(),
                    ].to_vec();

                    let triangulation = shape.to_triangulation(Some(FillRule::NonZero), 0.0);

                    let mut index = 0;
                    let mut real_tri_vertices : Vec<PointVertex> = Vec::new();
                    for tri in triangulation.points {
                        let mut array : [f32; 3] = [0.0, 0.0, 0.0];
                        array[index] = 1.0;
                        real_tri_vertices.push(
                            PointVertex {
                                position : [tri.x as f32, tri.y as f32, 0.0],
                                value : array,
                            }
                        );
                        index += 1;
                        index = index % 3;
                    }

                    let mut tri_indices : Vec<u16> = Vec::new();
                    for index in triangulation.indices {
                        tri_indices.push(index as u16);
                    }

                    let indices_2_len = tri_indices.len();

                    let vertex_buffer_2 = self.device.create_buffer_init(
                        &wgpu::util::BufferInitDescriptor {
                            label: Some("Vertex Buffer"),
                            contents: bytemuck::cast_slice(real_tri_vertices.as_slice()),
                            usage: wgpu::BufferUsages::VERTEX,
                        }
                    );

                    let index_buffer_2 = self.device.create_buffer_init(
                        &wgpu::util::BufferInitDescriptor {
                            label: Some("Index Buffer"),
                            contents: bytemuck::cast_slice(tri_indices.as_slice()),
                            usage: wgpu::BufferUsages::INDEX,
                        }
                    );

                    render_pass.set_pipeline(&self.render_pipeline_2); // 2.
                    render_pass.set_bind_group(0, &self.camera_bind_group, &[]); // NEW!
                    render_pass.set_vertex_buffer(0, vertex_buffer_2.slice(..));
                    render_pass.set_index_buffer(index_buffer_2.slice(..), wgpu::IndexFormat::Uint16);
                    // render_pass.draw(0..tris.len() as u32, 0..1);
                    render_pass.draw_indexed(0..indices_2_len as u32, 0, 0..1); // 2.
                }
            }
        }

        // submit will accept anything that implements IntoIter
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}
