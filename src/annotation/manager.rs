
use crate::PointVertex;
use crate::LineVertex;

use geo::Intersects;
use geo::Polygon;

#[derive(Clone, Debug)]
pub struct PointList {
    points : Vec<geo::Coord<f32>>,
}

pub struct Label {
    pub point_list : PointList,
}

pub struct LabelManager {
    labels : Vec<Label>,
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
        }
    }

    pub fn add_point(&mut self, x : f32, y : f32) {
        let new_point : geo::Coord<f32> = (x, y).into();

        self.points.push(new_point);
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

}


impl Label {
    pub fn new() -> Label {
        Label {
            point_list : PointList::new(),
        }
    }

    pub fn clear(&mut self) {
        self.point_list.clear();
    }

    pub fn to_geo_polygon(&self) -> Option<geo::Polygon<f32>> {
        self.point_list.to_geo_polygon()
    }

    pub fn get_draw_line(&self, zoom : f32) -> Option<(Vec<LineVertex>, Vec<u16>)> {
        self.point_list.get_draw_line(zoom)
    }

    pub fn to_point_vertices(&self) -> Option<Vec<PointVertex>> {
        self.point_list.to_point_vertices()
    }
}

impl LabelManager {
    pub fn new() -> LabelManager {
        LabelManager {
            labels : Vec::new(),
        }
    }

    pub fn get_new_label(&mut self) -> usize {
        let label = Label::new();
        self.labels.push(label);
        self.labels.len() - 1
    }

    pub fn get_labels(&self) -> &Vec<Label> {
        return &self.labels;
    }

    pub fn get_label_at_index(&mut self, index : usize) -> &mut Label {
        return &mut self.labels[index];
    }

    pub fn get_poly_at_point(&mut self, point : geo::Coord<f32>) -> Option<usize> {
        let result : Vec<(usize, &Label)> = self.labels.iter()
            .enumerate()
            .filter(|&(_index, x)| match x.to_geo_polygon(){
                Some(poly) => poly.intersects(&point),
                None => false,
            })
            .collect();

        // if
        if result.len() != 0 {
            return Some(result[0].0);
        }

        return None;
    }
    // pub fn get_polygon_draw(&self) -> Option<Vec<PointVertex>>
}
// struct Anno
