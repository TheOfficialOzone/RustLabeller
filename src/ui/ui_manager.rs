

struct Centimeters(f32);

enum ScreenCoord {
    Pixel(u32),
    PercentageShortest(f32),
    PercentageHorizontal(f32),
    PercentageVertical(f32),
    Distance(Centimeters), // Assumed to be in CMs
}

// Returns the pixel according to the screen coord
fn screen_coord_to_pixel(screen_coord : ScreenCoord) -> u32 {
    match screen_coord {
        Pixel(pixel) => { return pixel },
        PercentageShortest(percentage) => {

        },
        _ => { 0 }
    }
}

struct UIPosition {
    top_left : ScreenCoord,
    top_right : ScreenCoord,
    width : ScreenCoord,
    height : ScreenCoord,
}

trait UIElement {
    // fn propagate_click();

    fn render();
}

struct UIManager {
    elements : Vec<UIElement>
}


impl UIManager {
    fn render() {
        for element in elements {
            element.render();
        }
    }
}

// Should always remain between 0 and 1
struct Roundness(f32);

struct UIButton {
    position : UIPosition,
    roundness : Roundness,
}

impl UIElement for UIButton {
    fn render() {

    }
}
