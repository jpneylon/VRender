#ifndef __VR_WINDOW_CLASS_H__
#define __VR_WINDOW_CLASS_H__

#include <gtkmm.h>

#include <time.h>
#include "VRender.h"

#define STARTDIR "/home/anand/code/data"
#define MAX_VOLUME_SIDE 256
#define MASK_BONUS 16
#define MAX_RADIUS 1


class VR_Window : public Gtk::Box
{
  public:
    VR_Window();
    virtual ~VR_Window();

    void open_file();
    void print_file();
    char *get_file_name() { return point_cloud_list_file; };

    void create_render_window();

  private:

    void select_file();
    void update_render_buffer();
    void set_render_density();
    void set_render_brightness();
    void set_render_offset();
    void set_render_scale();
    void update_render_zoom(gdouble x, gdouble y);
    void update_render_translation(gdouble x, gdouble y);
    void update_render_rotation(gdouble x, gdouble y);
    virtual bool render_button_press_event(GdkEventButton *event);
    virtual bool render_motion_notify_event(GdkEventMotion *event);

    VRender *vrender;
    Cloud *cloud;

    float3 volume_origin;

    char *point_cloud_list_file;
    bool pc_file_open;
    bool adaptive_world_sizing;

    Gtk::Image      render_image;
    Gtk::Label      fps_update;
    Gtk::Label      fps_label;
    Gtk::Box        fps_box;

    Glib::RefPtr<Gtk::Adjustment>     dens_adjust;
    Glib::RefPtr<Gtk::Adjustment>     bright_adjust;
    Glib::RefPtr<Gtk::Adjustment>     offset_adjust;
    Glib::RefPtr<Gtk::Adjustment>     scale_adjust;
};






#endif // __VR_WINDOW_CLASS_H__
