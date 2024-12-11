#![allow(warnings)]

use std::sync::Arc;

use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, Event, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use RustMusicPlayer::State;


struct App<'a> {
    state: Option<State<'a>>,

    window: Option<Arc<Window>>,
}

impl ApplicationHandler for App<'_> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(event_loop
            .create_window(Window::default_attributes())
            .unwrap());
        window.set_min_inner_size(Some(LogicalSize::new(400.0, 400.0)));
        self.window = Some(window.clone());

        self.state = Some(pollster::block_on(State::new(window.clone())));
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        self.window.as_mut().unwrap().request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested
            | WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::Escape),
                        ..
                    },
                ..
            } => {
                event_loop.exit();
            },
            WindowEvent::Resized(physical_size) => {
                self.state.as_mut().unwrap().resize(physical_size);
                // let _ = self.state.as_mut().unwrap().render();
            },
            WindowEvent::RedrawRequested => {
                self.state.as_mut().unwrap().update();
                let _ = self.state.as_mut().unwrap().render(self.window.as_ref().unwrap());
            },
            _ => {}
        }
        match self.state.as_mut() {
            None => (),
            Some(state) => {state.input(self.window.as_ref().unwrap(), &event);}
        }
    }
}

fn run() {
    let event_loop = EventLoop::new().unwrap();

    // ControlFlow::Poll continuously runs the event loop, even if the OS hasn't
    // dispatched any events. This is ideal for games and similar applications.
    // event_loop.set_control_flow(ControlFlow::Poll);

    // ControlFlow::Wait pauses the event loop if no events are available to process.
    // This is ideal for non-game applications that only update in response to user
    // input, and uses significantly less power/CPU time than ControlFlow::Poll.
    // event_loop.set_control_flow(ControlFlow::Wait);
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App {
        state: None,
        window: None
    };

    let _ = event_loop.run_app(&mut app);
}

fn main() {
    run();
}
