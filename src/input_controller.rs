

pub mod Mouse {
    use std::sync::atomic::{self, Ordering::Relaxed};

    #[derive(PartialEq, Clone, Copy, Debug)]
    pub enum MouseState {
        Pressed,
        Released,
    }



    #[derive(PartialEq, Clone, Copy, Debug, Default)]
    pub struct MousePosition {
        pub x : isize,
        pub y : isize,
    }

    impl std::ops::Sub for MousePosition {
        type Output = MousePosition;
        fn sub(self, rhs: Self) -> Self::Output {
            let mut new_pos = MousePosition::default();
            new_pos.x = self.x - rhs.x;
            new_pos.y = self.y - rhs.y;

            return new_pos;
        }
    }

    static MOUSE_POS : std::sync::Mutex<MousePosition> = std::sync::Mutex::new(MousePosition{x: 0, y: 0});
    static MOUSE_STATE : atomic::AtomicBool = atomic::AtomicBool::new(false);
    static RMB_STATE : atomic::AtomicBool = atomic::AtomicBool::new(false);
    static MMB_STATE : atomic::AtomicBool = atomic::AtomicBool::new(false);

    pub fn set_mouse_pos(pos : MousePosition) {
        match MOUSE_POS.lock() {
            Ok(mut data) => *data = pos,
            Err(_) => (),
        }
    }

    pub fn get_mouse_pos() -> MousePosition {
        let mouse_pos = MOUSE_POS.lock().unwrap();
        let copy : MousePosition = *mouse_pos;
        return copy.clone();
    }

    pub fn set_mouse_down() {
        MOUSE_STATE.store(true, Relaxed);
    }

    pub fn set_mouse_up() {
        MOUSE_STATE.store(false, Relaxed);
    }

    pub fn set_mmb_down() {
        MMB_STATE.store(true, Relaxed);
    }

    pub fn set_mmb_up() {
        MMB_STATE.store(false, Relaxed);
    }

    // Gets the mouses state
    pub fn get_MMB_state() -> MouseState {
        let is_down = MMB_STATE.load(Relaxed);

        match is_down {
            true => MouseState::Pressed,
            false => MouseState::Released,
        }
    }

    pub fn set_RMB_down() {
        RMB_STATE.store(true, Relaxed);
    }

    pub fn set_RMB_up() {
        RMB_STATE.store(false, Relaxed);
    }

    // Gets the mouses state
    pub fn get_RMB_state() -> MouseState {
        let is_down = RMB_STATE.load(Relaxed);

        match is_down {
            true => MouseState::Pressed,
            false => MouseState::Released,
        }
    }

    // Gets the mouses state
    pub fn get_mouse_state() -> MouseState {
        let is_down = MOUSE_STATE.load(Relaxed);

        match is_down {
            true => MouseState::Pressed,
            false => MouseState::Released,
        }
    }

}
