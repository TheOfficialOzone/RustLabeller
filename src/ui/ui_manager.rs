// use crate::state;

use wgpu::util::RenderEncoder;

use crate::{global::{self, state::{EditState, EDIT_STATE}}, PointVertex};
use std::sync::atomic::Ordering::Relaxed;

struct Centimeters(f32);

pub enum ScreenCoord {
    Pixel(u32),
    PercentageShortest(f32),
    PercentageHorizontal(f32),
    PercentageVertical(f32),
    Distance(Centimeters), // Assumed to be in CMs
}

impl ScreenCoord {
    fn to_pixel(&self) -> u32 {
        match self {
            ScreenCoord::Pixel(pixel) => { return *pixel; },
            ScreenCoord::PercentageShortest(percentage) => {
                return 0;
            },
            _ => { 0 }
        }
    }
}


pub struct UIArea {
    pub top_edge: ScreenCoord,
    pub left_edge : ScreenCoord,
    pub width : ScreenCoord,
    pub height : ScreenCoord,
}

struct WGPUArea {
    pub top : f32,
    pub left : f32,
    pub width : f32,
    pub height : f32,
}

impl UIArea {
    /// Converts the area to the graphics rendering coordinate system
    /// Outputs in the order of [TOP, LEFT, WIDTH, HEIGHT]
    fn to_wgpu_coords(&self) -> WGPUArea {
        let mut window_width : f32 = global::state::WINDOW_SIZE_X.load(std::sync::atomic::Ordering::Relaxed) as f32;
        let mut window_height : f32 = global::state::WINDOW_SIZE_Y.load(std::sync::atomic::Ordering::Relaxed) as f32;

        // Adjust the window width for WGPU Coordinates
        window_width = window_width / 2.0;
        window_height = window_height / 2.0;

        WGPUArea {
            top:  1.0 - self.top_edge.to_pixel()  as f32 / window_height,
            left: -1.0 + self.left_edge.to_pixel() as f32 / window_width,
            width: self.width.to_pixel() as f32 / window_width,
            height: 0.0 - self.height.to_pixel() as f32 / window_height,
        }
    }

    fn in_bounds(&self, x : u32, y : u32) -> bool {
        let top = self.top_edge.to_pixel();
        let bottom = top + self.height.to_pixel();
        let left = self.left_edge.to_pixel();
        let right = top + self.width.to_pixel();

        return top < y && y < bottom && left < x && x < right;
    }

    /// Pushes the data to a buffer to render
    /// Bad design LOL
    fn push_to_buffer(&self, vertices : &mut Vec<PointVertex>, indices : &mut Vec<u16>) {
        let wgpu_pos = self.to_wgpu_coords();

        let top_left_vertex = PointVertex {
            position: [wgpu_pos.left, wgpu_pos.top, 0.0],
            value: [1.0, 0.0, 0.0],
        };
        let top_right_vertex = PointVertex {
            position: [wgpu_pos.left + wgpu_pos.width, wgpu_pos.top, 0.0],
            value: [1.0, 0.0, 0.0],
        };
        let bottom_left_vertex = PointVertex {
            position: [wgpu_pos.left, wgpu_pos.top + wgpu_pos.height, 0.0],
            value: [1.0, 0.0, 0.0],
        };
        let bottom_right_vertex = PointVertex {
            position: [wgpu_pos.left + wgpu_pos.width, wgpu_pos.top + wgpu_pos.height, 0.0],
            value: [1.0, 0.0, 0.0],
        };

        let start_index : u16 = vertices.len() as u16;

        vertices.push(top_left_vertex);
        vertices.push(top_right_vertex);
        vertices.push(bottom_left_vertex);
        vertices.push(bottom_right_vertex);

        indices.push(start_index + 0);
        indices.push(start_index + 1);
        indices.push(start_index + 2);
        indices.push(start_index + 1);
        indices.push(start_index + 2);
        indices.push(start_index + 3);
    }
}


pub struct UIManager {
    pub elements : Vec<Box<dyn UIElement>>,
    // elements : Vec<UIElement>,
    //
    pub render_pipeline : wgpu::RenderPipeline,

    pub vertex_buffer : wgpu::Buffer,
    pub index_buffer : wgpu::Buffer,
}


pub trait UIElement {
    fn check_click(&self, mouse_x : u32, mouse_y : u32) -> bool;

    fn click(&self, mouse_x : u32, mouse_y : u32);

    fn render(&self, points : &mut Vec<PointVertex>, indices : &mut Vec<u16>);
}

impl UIManager {
    /// Propogates a click to the U.I.
    /// Returns true if it was captured by the UI
    pub fn propagate_click(&self, mouse_x : u32, mouse_y : u32) -> bool {
        for element in self.elements.iter() {
            if element.check_click(mouse_x, mouse_y) {
                element.click(mouse_x, mouse_y);
                return true;
            }
        }
        return false;
    }

    pub fn render(&self, render_pass : &mut wgpu::RenderPass, queue : &mut wgpu::Queue) {
        render_pass.set_pipeline(&self.render_pipeline);

        let mut points : Vec<PointVertex> = Vec::new();
        let mut indices : Vec<u16> = Vec::new();
        for element in self.elements.iter() {
            element.render(&mut points, &mut indices);
        };

        // render_pass.buf
        queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(points.as_slice()));
        queue.write_buffer(&self.index_buffer, 0, bytemuck::cast_slice(indices.as_slice()));
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        render_pass.draw_indexed(0..indices.len() as u32, 0, 0..1); // 2.
    }
}

// Should always remain between 0 and 1
struct Roundness(f32);



pub struct EditButton {
    pub area : UIArea,
}

impl UIElement for EditButton {
    fn check_click(&self, mouse_x : u32, mouse_y : u32) -> bool{
        self.area.in_bounds(mouse_x, mouse_y)
    }

    fn click(&self, mouse_x : u32, mouse_y : u32) {
        let current_state : EditState = crate::global::state::EDIT_STATE.load(Relaxed);

        match current_state {
            EditState::New => { EDIT_STATE.store(EditState::Subtract, Relaxed); println!("To Subtract"); },
            EditState::Additive => {},
            EditState::Subtract => { EDIT_STATE.store(EditState::Select, Relaxed); println!("To Select"); },
            EditState::Select => { EDIT_STATE.store(EditState::New, Relaxed); println!("To New"); },
        }
    }

    fn render(&self, points : &mut Vec<PointVertex>, indices : &mut Vec<u16>) {
        self.area.push_to_buffer(points, indices);
    }
}



pub struct UIButton {
    pub area : UIArea,
    // roundness : Roundness,
    pub function : Box<dyn Fn()>,
}

impl UIElement for UIButton {
    fn check_click(&self, mouse_x : u32, mouse_y : u32) -> bool{
        self.area.in_bounds(mouse_x, mouse_y)
    }

    fn click(&self, mouse_x : u32, mouse_y : u32) {
        (self.function)()
    }

    fn render(&self, points : &mut Vec<PointVertex>, indices : &mut Vec<u16>) {
        self.area.push_to_buffer(points, indices);
    }
}





pub fn setup_ui(device : &wgpu::Device, config : &wgpu::SurfaceConfiguration) -> UIManager {
    // Shader 4
    let shader_4 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Button Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../button_shader.wgsl").into()),
    });

    let render_pipeline_layout_4 = device.create_pipeline_layout(
        &wgpu::PipelineLayoutDescriptor {
            label: Some("Button Render Pipeline Layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        }
    );

    let render_pipeline_4 = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Render Pipeline 4"),
        layout: Some(&render_pipeline_layout_4),
        vertex: wgpu::VertexState {
            module: &shader_4,
            entry_point: "vs_main", // 1.
            buffers: &[
                PointVertex::desc(),
            ],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState { // 3.
            module: &shader_4,
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

    let mut button_vec : Vec<Box<dyn UIElement>> = Vec::new();
    button_vec.push(
        Box::new(EditButton {
            area : UIArea {
                top_edge : ScreenCoord::Pixel(10),
                left_edge : ScreenCoord::Pixel(0),
                width : ScreenCoord::Pixel(500),
                height : ScreenCoord::Pixel(400),
            },
        })
        // Box::new(UIButton {
        //     area : UIArea {
        //         top_edge : ScreenCoord::Pixel(10),
        //         left_edge : ScreenCoord::Pixel(0),
        //         width : ScreenCoord::Pixel(500),
        //         height : ScreenCoord::Pixel(400),
        //     },
        //     function: Box::new(print_hey),
        // })
    );

    return UIManager {
        elements : button_vec,
        render_pipeline : render_pipeline_4,
        vertex_buffer : device.create_buffer(
            &wgpu::BufferDescriptor {
                label: Some("Button Vertex Buffer"),
                size: (std::mem::size_of::<PointVertex>() * 4) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation : false,
            }
        ),
        index_buffer : device.create_buffer(
            &wgpu::BufferDescriptor {
                label: Some("Button Index Buffer"),
                size: (std::mem::size_of::<u16>() * 6) as u64,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation : false,
            }
        )
    };
}
