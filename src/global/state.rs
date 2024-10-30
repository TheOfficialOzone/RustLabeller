use atomic_enum::atomic_enum;

#[atomic_enum]
#[derive(PartialEq)]
enum EditState {
    New = 0,
    Additive,
    Subtract,
    Split,
}

static EDIT_STATE : AtomicEditState = AtomicEditState::new(EditState::New);
// static EDIT_STATE : std::sync::Mutex<EditState> = std::sync::Mutex::new(EditState { NEW });
