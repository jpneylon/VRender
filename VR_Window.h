#ifndef __VR_WINDOW_CLASS_H__
#define __VR_WINDOW_CLASS_H__

#include <gtkmm.h>
#include <vector>
#include "VRender.h"

#define STARTDIR "/home/anand/code/data"
#define MAX_VOLUME_SIZE 512

class Cloud
{
  public:
    Cloud()
    {
        count = 0;
        max_pos.x = -999999;
        max_pos.y = -999999;
        max_pos.z = -999999;
        min_pos.x = 999999;
        min_pos.y = 999999;
        min_pos.z = 999999;
    }

    std::vector<float3> position;
    std::vector<uint3> rgb;
    uint count;
    float3 max_pos;
    float3 min_pos;
    float3 data_dim;
    float3 world_origin;
    float3 world_start;
    uint   world_size;
    float  world_res;
    uint   world_count;
    float  world_dim;
};


class VR_Window : public Gtk::Box
{
  public:
    VR_Window();
    virtual ~VR_Window();

    void open_file();
    void print_file();
    char *get_file_name() { return point_cloud_list_file; };

    void create_render_window();
    void destroy_render_window();
    bool get_renderer_open()
    {
        if (pc_file_open)
            return renderer_open;
        else
            return "No open files.";
    };

  private:

    void select_file();
    void create_color_maps();
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

    VRender vrender;
    Cloud cloud;

    bool maps_allocated;
    unsigned char *red_map;
    unsigned char *green_map;
    unsigned char *blue_map;

    char *point_cloud_list_file;
    bool renderer_open;
    bool pc_file_open;

    Gtk::Image               render_image;

    Glib::RefPtr<Gtk::Adjustment>     dens_adjust;
    Glib::RefPtr<Gtk::Adjustment>     bright_adjust;
    Glib::RefPtr<Gtk::Adjustment>     offset_adjust;
    Glib::RefPtr<Gtk::Adjustment>     scale_adjust;
};






#endif // __VR_WINDOW_CLASS_H__
