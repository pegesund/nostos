//! Glam linear algebra extension for Nostos.
//!
//! Provides vector and matrix operations using the glam library.
//! This is a test extension to validate the FFI architecture.

use nostos_extension::*;
use glam::{Vec2, Vec3, Vec4, Mat3, Mat4, Quat};

declare_extension!("glam", "0.1.0", register);

fn register(reg: &mut ExtRegistry) {
    // Vec2 operations
    reg.add("Glam.vec2", vec2_new);
    reg.add("Glam.vec2Add", vec2_add);
    reg.add("Glam.vec2Sub", vec2_sub);
    reg.add("Glam.vec2Mul", vec2_mul);
    reg.add("Glam.vec2Dot", vec2_dot);
    reg.add("Glam.vec2Length", vec2_length);
    reg.add("Glam.vec2Normalize", vec2_normalize);

    // Vec3 operations
    reg.add("Glam.vec3", vec3_new);
    reg.add("Glam.vec3Add", vec3_add);
    reg.add("Glam.vec3Sub", vec3_sub);
    reg.add("Glam.vec3Mul", vec3_mul);
    reg.add("Glam.vec3Dot", vec3_dot);
    reg.add("Glam.vec3Cross", vec3_cross);
    reg.add("Glam.vec3Length", vec3_length);
    reg.add("Glam.vec3Normalize", vec3_normalize);

    // Vec4 operations
    reg.add("Glam.vec4", vec4_new);
    reg.add("Glam.vec4Add", vec4_add);
    reg.add("Glam.vec4Dot", vec4_dot);

    // Mat4 operations
    reg.add("Glam.mat4Identity", mat4_identity);
    reg.add("Glam.mat4Translate", mat4_translate);
    reg.add("Glam.mat4Scale", mat4_scale);
    reg.add("Glam.mat4RotateX", mat4_rotate_x);
    reg.add("Glam.mat4RotateY", mat4_rotate_y);
    reg.add("Glam.mat4RotateZ", mat4_rotate_z);
    reg.add("Glam.mat4Mul", mat4_mul);
    reg.add("Glam.mat4MulVec4", mat4_mul_vec4);
    reg.add("Glam.mat4Perspective", mat4_perspective);
    reg.add("Glam.mat4LookAt", mat4_look_at);

    // Quaternion operations
    reg.add("Glam.quatFromAxisAngle", quat_from_axis_angle);
    reg.add("Glam.quatMul", quat_mul);
    reg.add("Glam.quatRotateVec3", quat_rotate_vec3);
}

// Helper to convert Vec2 to Value (tuple of floats)
fn vec2_to_value(v: Vec2) -> Value {
    Value::Tuple(std::sync::Arc::new(vec![
        Value::Float(v.x as f64),
        Value::Float(v.y as f64),
    ]))
}

// Helper to convert Value to Vec2
fn value_to_vec2(v: &Value) -> Result<Vec2, String> {
    match v {
        Value::Tuple(t) if t.len() == 2 => {
            let x = t[0].as_f32()?;
            let y = t[1].as_f32()?;
            Ok(Vec2::new(x, y))
        }
        Value::List(l) if l.len() == 2 => {
            let x = l[0].as_f32()?;
            let y = l[1].as_f32()?;
            Ok(Vec2::new(x, y))
        }
        _ => Err("Expected tuple or list of 2 floats for Vec2".to_string()),
    }
}

// Helper to convert Vec3 to Value
fn vec3_to_value(v: Vec3) -> Value {
    Value::Tuple(std::sync::Arc::new(vec![
        Value::Float(v.x as f64),
        Value::Float(v.y as f64),
        Value::Float(v.z as f64),
    ]))
}

// Helper to convert Value to Vec3
fn value_to_vec3(v: &Value) -> Result<Vec3, String> {
    match v {
        Value::Tuple(t) if t.len() == 3 => {
            let x = t[0].as_f32()?;
            let y = t[1].as_f32()?;
            let z = t[2].as_f32()?;
            Ok(Vec3::new(x, y, z))
        }
        Value::List(l) if l.len() == 3 => {
            let x = l[0].as_f32()?;
            let y = l[1].as_f32()?;
            let z = l[2].as_f32()?;
            Ok(Vec3::new(x, y, z))
        }
        _ => Err("Expected tuple or list of 3 floats for Vec3".to_string()),
    }
}

// Helper to convert Vec4 to Value
fn vec4_to_value(v: Vec4) -> Value {
    Value::Tuple(std::sync::Arc::new(vec![
        Value::Float(v.x as f64),
        Value::Float(v.y as f64),
        Value::Float(v.z as f64),
        Value::Float(v.w as f64),
    ]))
}

// Helper to convert Value to Vec4
fn value_to_vec4(v: &Value) -> Result<Vec4, String> {
    match v {
        Value::Tuple(t) if t.len() == 4 => {
            let x = t[0].as_f32()?;
            let y = t[1].as_f32()?;
            let z = t[2].as_f32()?;
            let w = t[3].as_f32()?;
            Ok(Vec4::new(x, y, z, w))
        }
        Value::List(l) if l.len() == 4 => {
            let x = l[0].as_f32()?;
            let y = l[1].as_f32()?;
            let z = l[2].as_f32()?;
            let w = l[3].as_f32()?;
            Ok(Vec4::new(x, y, z, w))
        }
        _ => Err("Expected tuple or list of 4 floats for Vec4".to_string()),
    }
}

// Helper to convert Mat4 to Value (list of 16 floats, column-major)
fn mat4_to_value(m: Mat4) -> Value {
    let cols = m.to_cols_array();
    Value::List(std::sync::Arc::new(
        cols.iter().map(|f| Value::Float(*f as f64)).collect()
    ))
}

// Helper to convert Value to Mat4
fn value_to_mat4(v: &Value) -> Result<Mat4, String> {
    let list = v.as_list()?;
    if list.len() != 16 {
        return Err("Expected list of 16 floats for Mat4".to_string());
    }
    let mut cols = [0.0f32; 16];
    for (i, val) in list.iter().enumerate() {
        cols[i] = val.as_f32()?;
    }
    Ok(Mat4::from_cols_array(&cols))
}

// Helper to convert Quat to Value
fn quat_to_value(q: Quat) -> Value {
    Value::Tuple(std::sync::Arc::new(vec![
        Value::Float(q.x as f64),
        Value::Float(q.y as f64),
        Value::Float(q.z as f64),
        Value::Float(q.w as f64),
    ]))
}

// Helper to convert Value to Quat
fn value_to_quat(v: &Value) -> Result<Quat, String> {
    match v {
        Value::Tuple(t) if t.len() == 4 => {
            let x = t[0].as_f32()?;
            let y = t[1].as_f32()?;
            let z = t[2].as_f32()?;
            let w = t[3].as_f32()?;
            Ok(Quat::from_xyzw(x, y, z, w))
        }
        _ => Err("Expected tuple of 4 floats for Quat".to_string()),
    }
}

// ==================== Vec2 Operations ====================

fn vec2_new(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let x = args[0].as_f32()?;
    let y = args[1].as_f32()?;
    Ok(vec2_to_value(Vec2::new(x, y)))
}

fn vec2_add(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let a = value_to_vec2(&args[0])?;
    let b = value_to_vec2(&args[1])?;
    Ok(vec2_to_value(a + b))
}

fn vec2_sub(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let a = value_to_vec2(&args[0])?;
    let b = value_to_vec2(&args[1])?;
    Ok(vec2_to_value(a - b))
}

fn vec2_mul(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let v = value_to_vec2(&args[0])?;
    let s = args[1].as_f32()?;
    Ok(vec2_to_value(v * s))
}

fn vec2_dot(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let a = value_to_vec2(&args[0])?;
    let b = value_to_vec2(&args[1])?;
    Ok(Value::Float(a.dot(b) as f64))
}

fn vec2_length(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let v = value_to_vec2(&args[0])?;
    Ok(Value::Float(v.length() as f64))
}

fn vec2_normalize(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let v = value_to_vec2(&args[0])?;
    Ok(vec2_to_value(v.normalize()))
}

// ==================== Vec3 Operations ====================

fn vec3_new(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let x = args[0].as_f32()?;
    let y = args[1].as_f32()?;
    let z = args[2].as_f32()?;
    Ok(vec3_to_value(Vec3::new(x, y, z)))
}

fn vec3_add(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let a = value_to_vec3(&args[0])?;
    let b = value_to_vec3(&args[1])?;
    Ok(vec3_to_value(a + b))
}

fn vec3_sub(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let a = value_to_vec3(&args[0])?;
    let b = value_to_vec3(&args[1])?;
    Ok(vec3_to_value(a - b))
}

fn vec3_mul(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let v = value_to_vec3(&args[0])?;
    let s = args[1].as_f32()?;
    Ok(vec3_to_value(v * s))
}

fn vec3_dot(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let a = value_to_vec3(&args[0])?;
    let b = value_to_vec3(&args[1])?;
    Ok(Value::Float(a.dot(b) as f64))
}

fn vec3_cross(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let a = value_to_vec3(&args[0])?;
    let b = value_to_vec3(&args[1])?;
    Ok(vec3_to_value(a.cross(b)))
}

fn vec3_length(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let v = value_to_vec3(&args[0])?;
    Ok(Value::Float(v.length() as f64))
}

fn vec3_normalize(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let v = value_to_vec3(&args[0])?;
    Ok(vec3_to_value(v.normalize()))
}

// ==================== Vec4 Operations ====================

fn vec4_new(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let x = args[0].as_f32()?;
    let y = args[1].as_f32()?;
    let z = args[2].as_f32()?;
    let w = args[3].as_f32()?;
    Ok(vec4_to_value(Vec4::new(x, y, z, w)))
}

fn vec4_add(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let a = value_to_vec4(&args[0])?;
    let b = value_to_vec4(&args[1])?;
    Ok(vec4_to_value(a + b))
}

fn vec4_dot(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let a = value_to_vec4(&args[0])?;
    let b = value_to_vec4(&args[1])?;
    Ok(Value::Float(a.dot(b) as f64))
}

// ==================== Mat4 Operations ====================

fn mat4_identity(_args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    Ok(mat4_to_value(Mat4::IDENTITY))
}

fn mat4_translate(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let v = value_to_vec3(&args[0])?;
    Ok(mat4_to_value(Mat4::from_translation(v)))
}

fn mat4_scale(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let v = value_to_vec3(&args[0])?;
    Ok(mat4_to_value(Mat4::from_scale(v)))
}

fn mat4_rotate_x(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let angle = args[0].as_f32()?;
    Ok(mat4_to_value(Mat4::from_rotation_x(angle)))
}

fn mat4_rotate_y(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let angle = args[0].as_f32()?;
    Ok(mat4_to_value(Mat4::from_rotation_y(angle)))
}

fn mat4_rotate_z(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let angle = args[0].as_f32()?;
    Ok(mat4_to_value(Mat4::from_rotation_z(angle)))
}

fn mat4_mul(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let a = value_to_mat4(&args[0])?;
    let b = value_to_mat4(&args[1])?;
    Ok(mat4_to_value(a * b))
}

fn mat4_mul_vec4(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let m = value_to_mat4(&args[0])?;
    let v = value_to_vec4(&args[1])?;
    Ok(vec4_to_value(m * v))
}

fn mat4_perspective(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let fov_y = args[0].as_f32()?;
    let aspect = args[1].as_f32()?;
    let z_near = args[2].as_f32()?;
    let z_far = args[3].as_f32()?;
    Ok(mat4_to_value(Mat4::perspective_rh(fov_y, aspect, z_near, z_far)))
}

fn mat4_look_at(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let eye = value_to_vec3(&args[0])?;
    let center = value_to_vec3(&args[1])?;
    let up = value_to_vec3(&args[2])?;
    Ok(mat4_to_value(Mat4::look_at_rh(eye, center, up)))
}

// ==================== Quaternion Operations ====================

fn quat_from_axis_angle(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let axis = value_to_vec3(&args[0])?;
    let angle = args[1].as_f32()?;
    Ok(quat_to_value(Quat::from_axis_angle(axis, angle)))
}

fn quat_mul(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let a = value_to_quat(&args[0])?;
    let b = value_to_quat(&args[1])?;
    Ok(quat_to_value(a * b))
}

fn quat_rotate_vec3(args: &[Value], _ctx: &ExtContext) -> Result<Value, String> {
    let q = value_to_quat(&args[0])?;
    let v = value_to_vec3(&args[1])?;
    Ok(vec3_to_value(q * v))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec3_operations() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let ctx = ExtContext::new(rt.handle().clone(), tx, Pid(1));

        // Create two vectors
        let v1 = vec3_new(&[Value::Float(1.0), Value::Float(2.0), Value::Float(3.0)], &ctx).unwrap();
        let v2 = vec3_new(&[Value::Float(4.0), Value::Float(5.0), Value::Float(6.0)], &ctx).unwrap();

        // Add them
        let sum = vec3_add(&[v1.clone(), v2.clone()], &ctx).unwrap();
        let sum_vec = value_to_vec3(&sum).unwrap();
        assert_eq!(sum_vec, Vec3::new(5.0, 7.0, 9.0));

        // Dot product
        let dot = vec3_dot(&[v1.clone(), v2.clone()], &ctx).unwrap();
        assert_eq!(dot.as_f64().unwrap(), 32.0); // 1*4 + 2*5 + 3*6 = 32

        // Cross product
        let cross = vec3_cross(&[v1, v2], &ctx).unwrap();
        let cross_vec = value_to_vec3(&cross).unwrap();
        // (2*6 - 3*5, 3*4 - 1*6, 1*5 - 2*4) = (-3, 6, -3)
        assert_eq!(cross_vec, Vec3::new(-3.0, 6.0, -3.0));
    }

    #[test]
    fn test_mat4_operations() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let ctx = ExtContext::new(rt.handle().clone(), tx, Pid(1));

        // Create identity matrix
        let identity = mat4_identity(&[], &ctx).unwrap();
        let identity_mat = value_to_mat4(&identity).unwrap();
        assert_eq!(identity_mat, Mat4::IDENTITY);

        // Create translation matrix
        let trans = mat4_translate(&[Value::Tuple(std::sync::Arc::new(vec![
            Value::Float(1.0),
            Value::Float(2.0),
            Value::Float(3.0),
        ]))], &ctx).unwrap();
        let trans_mat = value_to_mat4(&trans).unwrap();
        assert_eq!(trans_mat, Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0)));
    }
}
