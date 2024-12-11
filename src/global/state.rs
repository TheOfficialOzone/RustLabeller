use std::sync::atomic::AtomicU32;

use atomic_enum::atomic_enum;

#[atomic_enum]
#[derive(PartialEq)]
pub enum EditState {
    New = 0,
    Additive,
    Subtract,
    Select,
}

pub static EDIT_STATE : AtomicEditState = AtomicEditState::new(EditState::New);

// Probably a better way to do this but...
pub static WINDOW_SIZE_X : AtomicU32 = AtomicU32::new(0);
pub static WINDOW_SIZE_Y : AtomicU32 = AtomicU32::new(0);
