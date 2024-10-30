

// Vertex shader
struct Camera {
    x : f32,
    y : f32,
    zoom : f32,
}

@group(0) @binding(0)
var<uniform> camera: Camera;


struct VertexInput {
    @location(0) position: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position:vec4<f32>,
}


@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;

    var model_position = model.position;
    model_position.x -= camera.x / 800;
    model_position.y -= camera.y / 600;

    model_position /= camera.zoom;

    out.clip_position = vec4<f32>(model_position, 1.0) ;
    return out;
}


@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(1, 1, 1, 1.0);
}
