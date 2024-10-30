

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
    @location(1) value: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(1) value : vec3<f32>,
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

    out.value = model.value;
    // out.value.x = model.position.x;
    // out.value.y = 0.0;
    // out.value.z = 0.0;

    out.clip_position = vec4<f32>(model_position, 1.0) ;
    return out;
}


// Fragment shader


@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // if in.value.y < 0.1 || in.value.x < 0.1 || in.value.z < 0.1 {// 
    //     return vec4<f32>(1, 0, 0, 0.5);
    //     // discard;
    // }

    // return vec4<f32>(in.value.x, in.value.y, in.value.z, 0.5);
    return vec4<f32>(0, 0, 1, 0.5);
}
 

/*
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(1 - i32(in_vertex_index)) * 0.5;
    let y = f32(i32(in_vertex_index & 1u) * 2 - 1) * 0.5;
    out.clip_position = vec4<f32>(x, y, 0.0, 1.0);
    return out;
}

// Fragment shader

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(0.3, 0.2, 0.1, 1.0);
}
*/
