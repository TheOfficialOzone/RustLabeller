
use crate::global::state::EditState;
use crate::PointVertex;
use crate::LineVertex;
use crate::Selection;

use geo::BooleanOps;
use geo::CoordsIter;
use geo::MultiPolygon;
use i_triangle::{i_overlay::{core::fill_rule::FillRule, i_float::{f32_point::F32Point, f64_point::F64Point}}, triangulation::float::FloatTriangulate};
use geo::Intersects;
use geo::Polygon;
use wgpu::{util::DeviceExt, MemoryHints};


// enum ValidLabel {
//     INCOMPLETE,

// }

#[derive(Clone, Debug)]
pub struct PointList {
    points : Vec<geo::Coord<f32>>,
    mouse_point : Option<geo::Coord<f32>>,
}

pub struct Label {
    // pub point_list : PointList,
    pub polys : geo::MultiPolygon<f32>,
}


trait Normalizable {
    fn length(&self) -> f32;

    fn dot_product(&self, other : &geo::Coord<f32>) -> f32;

    fn zeroed(&self) -> bool;

    fn normalize(&mut self);

    fn angle(&self) -> f32;

    fn reflected_angle(&self) -> f32;
}

impl Normalizable for geo::Coord<f32> {
    fn length(&self) -> f32 {
        return (self.x.powf(2.0) + self.y.powf(2.0)).sqrt();
    }

    fn dot_product(&self, other : &geo::Coord<f32>) -> f32 {
        return self.x * other.x + self.y * other.y;
    }

    fn zeroed(&self) -> bool {
        return self.length() == 0.0;
    }

    fn normalize(&mut self) {
        if self.length() == 0.0 {
            return;
        }
        let length = self.length();
        self.x /= length;
        self.y /= length;
    }

    fn angle(&self) -> f32 {
        return self.y.atan2(self.x);
    }

    fn reflected_angle(&self) -> f32 {
        return self.x.atan2(self.y);
    }
}

impl PointList {
    pub fn new() -> PointList {
        PointList {
            points : [].to_vec(),
            mouse_point : None,
        }
    }

    pub fn add_point(&mut self, x : f32, y : f32) {
        let new_point : geo::Coord<f32> = (x, y).into();

        self.points.push(new_point);
    }

    pub fn add_mouse_point(&mut self, x : f32, y : f32) {
        self.mouse_point = Some((x,y).into());
    }

    pub fn remove_mouse_point(&mut self) {
        self.mouse_point = None;
    }

    fn length(&self) -> usize {
        return self.points.len();
    }

    pub fn to_point_vertices(&self) -> Option<Vec<PointVertex>> {
        if self.length() < 3 {
            return None;
        }

        let mut vertices : Vec<PointVertex> = Vec::with_capacity(self.length());

        for point in self.points.iter() {
            let new_point = PointVertex {
                position : [point.x, point.y, 0.0],
                value : [1.0, 0.0, 0.0],
            };
            vertices.push(new_point);
        }

        if let Some(mouse_point) = self.mouse_point {
            vertices.push(
                PointVertex { position: [mouse_point.x, mouse_point.y, 0.0], value: [1.0, 0.0, 0.0] }
            )
        }

        return Some(vertices);
    }

    pub fn get_draw_line(&self, scale : f32) -> Option<(Vec<LineVertex>, Vec<u16>)> {
        if self.length() < 3 { return None; }

        let mut points : Vec<LineVertex> = Vec::new();
        let mut indices : Vec<u16> = Vec::new();

        for (index, point) in self.points.iter().enumerate() {
            // Get the previous & next point
            let mut prev_index : isize = index as isize - 1;
            if prev_index < 0 { prev_index += self.length() as isize; }

            let mut next_index = index + 1;
            if next_index >= self.length() { next_index -= self.length(); }

            let prev_point = self.points[prev_index as usize];
            let next_point = self.points[next_index];

            let mut adjust_vector = *point - prev_point;
            let mut prev_vector = prev_point - *point;
            let mut next_vector = next_point - *point;

            adjust_vector.normalize();
            prev_vector.normalize();
            next_vector.normalize();

            let bfor_cos = prev_vector.dot_product(&next_vector).clamp(-1.0, 1.0);
            let angle = f32::acos(bfor_cos);

            // let mut total_angle = (2.0 * 3.141592 - adjust_vector.angle()) + angle / 2.0;
            let mut total_angle = adjust_vector.reflected_angle();

            // Add PI / 2 to prev vector
            // Use dot product to determine if they are facing the same direciton
            let flip_angle = adjust_vector.angle() - 3.141592 / 2.0;
            let flip_vector : geo::Coord<f32> = (
                f32::cos(flip_angle),
                f32::sin(flip_angle),
            ).into();

            let flip = flip_vector.dot_product(&next_vector) > 0.0;

            match flip {
                true => {
                    total_angle -= angle / 2.0;
                },
                false => {
                    total_angle += angle / 2.0;
                }
            }

            // Gets the length
            let line_perp_angle = next_vector.reflected_angle() + 3.141592 / 2.0;
            let length = scale * 0.02 / f32::cos(line_perp_angle - total_angle);

            let middle_vec = [
                f32::sin(total_angle) * length, f32::cos(total_angle) * length
            ];

            let inner : LineVertex = LineVertex {
                position : [point.x - middle_vec[0], point.y - middle_vec[1], 0.0],
            };
            let outer : LineVertex = LineVertex {
                position : [point.x + middle_vec[0], point.y + middle_vec[1], 0.0],
            };

            points.push(inner);
            points.push(outer);
        }

        // GPU CODE THAT IS NOT IMPORTANT
        for index in 0..self.length() {
            let inner_index = index * 2;
            let outer_index = inner_index + 1;

            let next_inner_index = (inner_index + 2) % points.len();
            let next_outer_index = (inner_index + 3) % points.len();

            indices.push(inner_index as u16);
            indices.push(outer_index as u16);
            indices.push(next_inner_index as u16);

            indices.push(next_inner_index as u16);
            indices.push(next_outer_index as u16);
            indices.push(outer_index as u16);
        }

        return Some((points, indices));
    }

    pub fn clear(&mut self) {
        self.points.clear();
    }

    pub fn to_geo_polygon(&self) -> Option<geo::Polygon<f32>> {
        if self.length() < 3 {
            return None;
        }

        let points = self.points.clone();
        let line : geo::LineString<f32> = geo::LineString::from(points);

        let poly : geo::Polygon<f32> = geo::Polygon::new(
            line,
            vec![],
        );

        return Some(poly);
    }

    // Pushes itself onto the render buffer
    pub fn push_to_render_buffer(&self, vertices : &mut Vec<PointVertex>, indices : &mut Vec<u16>) {
        let polys_point = match self.to_point_vertices() {
            Some(points) => points,
            None => return,
        };

        // let mut real_tri_vertices : Vec<PointVertex> = Vec::new();
        // let mut tri_indices : Vec<u16> = Vec::new();

        // I_Triangle
        let tri_vertices : &[PointVertex] = &polys_point[0..polys_point.len()];
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


        let index_adjust : u16 = vertices.len() as u16;
        for tri in triangulation.points {
            vertices.push(
                PointVertex {
                    position : [tri.x as f32, tri.y as f32, 0.0],
                    value : [0.0, 0.0, 1.0],
                }
            );
        }

        // Index needs to be adjusted up
        for index in triangulation.indices {
            indices.push(index_adjust + index as u16);
        }
    }
}


// struct

// impl
#[derive(Default)]
struct VertexRenderBuffer {
    pub interiors : Vec<Vec<PointVertex>>,
    pub exterior : Vec<PointVertex>,
}

impl Label {
    pub fn new() -> Label {
        Label {
            polys: MultiPolygon::new([].to_vec()),
        }
    }

    pub fn get_draw_line(&self, zoom : f32) -> Option<(Vec<LineVertex>, Vec<u16>)> {
        return None;
        // self.point_list.get_draw_line(zoom)
    }

    // pub fn to_point_vertices(&self) -> Vec<Vec<PointVertex>> {
    pub fn to_point_vertices(&self) -> Vec<Vec<PointVertex>> {
        let mut label_vertices : Vec<Vec<PointVertex>> = Vec::with_capacity(self.polys.iter().count());

        for poly in self.polys.iter() {
            let mut poly_vertices : Vec<PointVertex> = Vec::new();

            for point in poly.exterior() {
                let new_point = PointVertex {
                    position : [point.x, point.y, 0.0],
                    value : [1.0, 0.0, 0.0],
                };
                poly_vertices.push(new_point);
            }
            label_vertices.push(poly_vertices);
        }

        return label_vertices;
    }

    fn to_vertex_render_buffer(&self) -> Vec<VertexRenderBuffer> {
        let mut label_vertices : Vec<VertexRenderBuffer> = Vec::with_capacity(self.polys.iter().count());

        for poly in self.polys.iter() {
            let mut vertex_buffer = VertexRenderBuffer::default();
            // let mut poly_exterior_vertices : Vec<PointVertex> = Vec::new();

            for point in poly.exterior() {
                let new_point = PointVertex {
                    position : [point.x, point.y, 0.0],
                    value : [1.0, 1.0, 0.0],
                };
                vertex_buffer.exterior.push(new_point);
            }

            for interior in poly.interiors() {
                let mut interior_vec : Vec<PointVertex> = Vec::new();
                for point in interior {
                    let new_point = PointVertex {
                        position : [point.x, point.y, 0.0],
                        value : [1.0, 1.0, 0.0],
                    };
                    interior_vec.push(new_point);
                }
                vertex_buffer.interiors.push(interior_vec);
            }

            label_vertices.push(vertex_buffer);
        }

        return label_vertices;
    }

    // Subtracts a poly from the shape
    pub fn subtract_poly(&mut self, poly : &geo::Polygon<f32>) {
        let multipolygon = MultiPolygon(vec![poly.clone()]);
        self.polys = self.polys.difference(&multipolygon);
    }

    // Adds a poly to another
    pub fn add_poly(&mut self, poly : &geo::Polygon<f32>) {
        let multipolygon = MultiPolygon(vec![poly.clone()]);
        self.polys = self.polys.union(&multipolygon);
    }

    // Pushes itself onto the render buffer
    pub fn push_to_render_buffer(&self, vertices : &mut Vec<PointVertex>, indices : &mut Vec<u16>, selected : bool) {
        let vertex_render_buffer = self.to_vertex_render_buffer();

        for buffer in vertex_render_buffer.iter() {
            let exterior = &buffer.exterior;
            let interiors = &buffer.interiors;

            // For I_Triangle
            let mut exterior_vec : Vec<F64Point> = Vec::new();
            let mut interiors_vec : Vec<Vec<F64Point>> = Vec::new();

            for point in exterior {
                exterior_vec.push(
                    F64Point { x: point.position[0] as f64, y: point.position[1] as f64 }
                )
            }

            for interior in interiors {
                let mut interior_vec : Vec<F64Point> = Vec::new();
                for point in interior {
                    interior_vec.push(
                        F64Point { x: point.position[0] as f64, y: point.position[1] as f64 }
                    )
                }
                interiors_vec.push(interior_vec);
            }

            let mut shape = [
                exterior_vec, // Body
            ].to_vec();
            for interior_vec in interiors_vec {
                shape.push(interior_vec);
            }

            let triangulation = shape.to_triangulation(Some(FillRule::NonZero), 0.0);

            let index_adjust : u16 = vertices.len() as u16;

            let color = match selected {
                true => {[1.0, 0.0, 1.0]},
                false => {[1.0, 0.0, 0.0]},
            };

            for tri in triangulation.points {
                vertices.push(
                    PointVertex {
                        position : [tri.x as f32, tri.y as f32, 0.0],
                        value : color,
                    }
                )
            }
            for index in triangulation.indices {
                indices.push(index_adjust + index as u16);
            }
        }
    }
}

pub struct LabelManager {
    labels : Vec<Label>,
    // subtract_label : Option<Label>,
    edit_label : PointList,

    polygon_pipeline : wgpu::RenderPipeline,
    outline_pipeline : wgpu::RenderPipeline,
}

impl LabelManager {

    pub fn get_edit_label(&mut self) -> &mut PointList {
        return &mut self.edit_label;
    }

    pub fn propagate_add_label(&mut self, polygon : geo::Polygon<f32>, selection : &Selection) {
        // It should directly access the values to remove them
        for (index, label) in self.labels.iter_mut().enumerate() {
            let selected = match selection {
                Selection::None => { false },
                Selection::Selected(x) => { index == *x },
                Selection::MultiSelection(values) => { values.contains(&index) },
            };
            if selected {
                label.add_poly(&polygon);
            }
        }
    }

    pub fn propagate_subtract_label(&mut self, polygon : geo::Polygon<f32>, selection : &Selection) {
        for (index, label) in self.labels.iter_mut().enumerate() {
            let selected = match selection {
                Selection::None => { false },
                Selection::Selected(x) => { index == *x },
                Selection::MultiSelection(values) => { values.contains(&index) },
            };
            if selected {
                label.subtract_poly(&polygon);
            }
        }
    }

    pub fn propagate_edit_label(&mut self, selection : &Selection, edit_state : &EditState) {
        let geo_poly = match self.edit_label.to_geo_polygon() {
            Some(geo_poly) => geo_poly,
            _ => return,
        };

        match edit_state {
            EditState::New => {
                let label : Label = Label { polys : MultiPolygon(vec![geo_poly])};
                self.labels.push(label);
            },
            EditState::Subtract => {
                self.propagate_subtract_label(geo_poly, &selection);
            },
            EditState::Additive => {
                self.propagate_add_label(geo_poly, &selection);
            },
            _ => {println!("No Function");},
        }

        self.edit_label.clear();
    }

    pub fn split_selected_labels(&mut self, selection : &Selection) {
        let selections = match selection {
            Selection::MultiSelection(selections) => selections,
            Selection::Selected(selected) => &[*selected].to_vec(),
            Selection::None => return,
        };

        let mut new_labels : Vec<Label> = Vec::new();

        for label_index in selections {
            if let Some(label) = self.labels.get(*label_index) {
                for poly in label.polys.iter() {
                    let mut new_label = Label::new();
                    new_label.add_poly(&poly);
                    new_labels.push(
                        new_label
                    );
                }
            }
        }

        // Remove the values that were selected
        let mut index = 0;
        self.labels.retain(|_| {
            index += 1;
            !selections.contains(&(index - 1))
        });

        // Adds the new labels
        for label in new_labels {
            self.labels.push(label);
        }
    }

    pub fn merge_selected_labels(&mut self, selection : &Selection) {
        // Only works with multiple selections
        let selections = match selection {
            Selection::MultiSelection(selections) => selections,
            _ => return,
        };

        let mut combined_label = Label::new();

        // The polygons we are selecting
        for label_index in selections {
            if let Some(label) = self.labels.get(*label_index) {
                for poly in &label.polys {
                    combined_label.add_poly(poly);
                }
            }
        }

        // Remove the values that were selected
        let mut index = 0;
        self.labels.retain(|_| {
            index += 1;
            !selections.contains(&(index - 1))
        });

        // Push the combined label
        self.labels.push(combined_label);
    }

    pub fn get_labels(&self) -> &Vec<Label> {
        return &self.labels;
    }

    pub fn get_poly_at_point(&self, point : geo::Coord<f32>) -> Option<usize> {
        let result : Vec<(usize, &Label)> = self.labels.iter()
            .enumerate()
            .filter(|&(_index, x)| x.polys.intersects(&point))
            .collect();

        if result.len() != 0 {
            return Some(result[0].0);
        }

        return None;
    }

    pub fn render(&self, device : &wgpu::Device, render_pass : &mut wgpu::RenderPass, queue: &mut wgpu::Queue, camera_bind_group : &wgpu::BindGroup, camera : & crate::Camera, selection : &Selection) {
        let labels = self.get_labels();
        // Draw the Lines
        for label in labels {
            if let Some((line_points, line_indices)) = label.get_draw_line(camera.zoom.0) {
                let vertex_buffer_3 = device.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some("Vertex Buffer"),
                        contents: bytemuck::cast_slice(line_points.as_slice()),
                        usage: wgpu::BufferUsages::VERTEX,
                    }
                );

                let index_buffer_3 = device.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some("Index Buffer"),
                        contents: bytemuck::cast_slice(line_indices.as_slice()),
                        usage: wgpu::BufferUsages::INDEX,
                    }
                );
                render_pass.set_pipeline(&self.outline_pipeline);
                render_pass.set_bind_group(0, &camera_bind_group, &[]);
                render_pass.set_vertex_buffer(0, vertex_buffer_3.slice(..));
                render_pass.set_index_buffer(index_buffer_3.slice(..), wgpu::IndexFormat::Uint16);
                render_pass.draw_indexed(0..line_indices.len() as u32, 0, 0..1); // 2.
            }
        }

        // Draws the selected labels vertices
        {
            let mut vertices : Vec<PointVertex> = Vec::new();
            let mut indices : Vec<u16> = Vec::new();

            for (index, label) in labels.iter().enumerate() {
                let selected = match selection {
                    Selection::None => { false },
                    Selection::Selected(x) => { index == *x },
                    Selection::MultiSelection(values) => { values.contains(&index) },
                };
                label.push_to_render_buffer(&mut vertices, &mut indices, selected);
            }

            let vertex_buffer_2 = device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("Vertex Buffer"),
                    contents: bytemuck::cast_slice(vertices.as_slice()),
                    usage: wgpu::BufferUsages::VERTEX,
                }
            );

            let index_buffer_2 = device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("Index Buffer"),
                    contents: bytemuck::cast_slice(indices.as_slice()),
                    usage: wgpu::BufferUsages::INDEX,
                }
            );

            render_pass.set_pipeline(&self.polygon_pipeline); // 2.
            render_pass.set_bind_group(0, &camera_bind_group, &[]); // NEW!
            render_pass.set_vertex_buffer(0, vertex_buffer_2.slice(..));
            render_pass.set_index_buffer(index_buffer_2.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(0..indices.len() as u32, 0, 0..1); // 2.
        }

        // Rendering the points
        for label in labels {
            break;
            let polys = label.to_point_vertices();

            let mut vertex_blocks : Vec<PointVertex> = Vec::new();
            let mut indices : Vec<u16> = Vec::new();

            for poly in polys {
                for tri in poly {
                    const OFFSET : f32 = 0.01;

                    let x_offsets = vec![OFFSET, OFFSET, -OFFSET, -OFFSET];
                    let y_offsets = vec![OFFSET, -OFFSET, OFFSET, -OFFSET];

                    for (x_off, y_off) in x_offsets.iter().zip(y_offsets.iter()) {
                        vertex_blocks.push(
                            PointVertex {
                                position: [tri.position[0] - x_off, tri.position[1] - y_off, 0.0],
                                value: [1.0, 0.0, 0.0],
                            }
                        )
                    }

                    let base_index = vertex_blocks.len() as u16;
                    indices.push(base_index);
                    indices.push(base_index + 1);
                    indices.push(base_index + 2);
                    indices.push(base_index + 1);
                    indices.push(base_index + 2);
                    indices.push(base_index + 3);
                }
            }

            let vertex_buffer_2 = device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("Vertex Buffer"),
                    contents: bytemuck::cast_slice(vertex_blocks.as_slice()),
                    usage: wgpu::BufferUsages::VERTEX,
                }
            );

            let index_buffer_2 = device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("Index Buffer"),
                    contents: bytemuck::cast_slice(indices.as_slice()),
                    usage: wgpu::BufferUsages::INDEX,
                }
            );

            render_pass.set_pipeline(&self.polygon_pipeline); // 2.
            render_pass.set_bind_group(0, &camera_bind_group, &[]); // NEW!
            render_pass.set_vertex_buffer(0, vertex_buffer_2.slice(..));
            render_pass.set_index_buffer(index_buffer_2.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(0..indices.len() as u32, 0, 0..1); // 2.
        }

        // Draws the current edit
        if self.edit_label.length() >= 3 {
            let mut vertices : Vec<PointVertex> = Vec::new();
            let mut indices : Vec<u16> = Vec::new();

            self.edit_label.push_to_render_buffer(&mut vertices, &mut indices);

            let indices_2_len = indices.len();

            let vertex_buffer_2 = device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("Vertex Buffer"),
                    contents: bytemuck::cast_slice(vertices.as_slice()),
                    usage: wgpu::BufferUsages::VERTEX,
                }
            );

            let index_buffer_2 = device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("Index Buffer"),
                    contents: bytemuck::cast_slice(indices.as_slice()),
                    usage: wgpu::BufferUsages::INDEX,
                }
            );
            render_pass.set_pipeline(&self.polygon_pipeline); // 2.
            render_pass.set_bind_group(0, &camera_bind_group, &[]); // NEW
            render_pass.set_vertex_buffer(0, vertex_buffer_2.slice(..));
            render_pass.set_index_buffer(index_buffer_2.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(0..indices_2_len as u32, 0, 0..1); // 2.
        }
    }
}

pub fn setup_label_manager(device : &wgpu::Device, config :&wgpu::SurfaceConfiguration, camera_bind_group_layout : &wgpu::BindGroupLayout) -> LabelManager {
    // Shader_2
    let shader_2 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Polygon Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../polygon_shader.wgsl").into()),
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
        source: wgpu::ShaderSource::Wgsl(include_str!("../line_shader.wgsl").into()),
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

    LabelManager {
        labels: Vec::new(),
        edit_label : PointList::new(),
        polygon_pipeline: render_pipeline_2,
        outline_pipeline: render_pipeline_3,
    }
}
