#include "VR_Window.h"



VR_Window::VR_Window()
{
    set_orientation( Gtk::ORIENTATION_HORIZONTAL );

    pc_file_open = false;

    vrender = new VRender;
    cloud = new Cloud;
}

VR_Window::~VR_Window()
{
    delete vrender;
    delete cloud;
    hide();
}


void
VR_Window::select_file()
{
    GtkWidget *file;

    file = gtk_file_chooser_dialog_new( "Point Cloud Renderer", NULL,
                                        GTK_FILE_CHOOSER_ACTION_OPEN,
                                        "CANCEL", GTK_RESPONSE_CANCEL,
                                        "OPEN", GTK_RESPONSE_ACCEPT,
                                        NULL);
    gtk_file_chooser_set_filename( GTK_FILE_CHOOSER (file), STARTDIR );

    if (gtk_dialog_run ( GTK_DIALOG (file)) == GTK_RESPONSE_ACCEPT)
    {
        point_cloud_list_file = gtk_file_chooser_get_filename (GTK_FILE_CHOOSER (file));
    }
    gtk_widget_destroy (file);
}
void
VR_Window::open_file()
{
    ///////////////////////////////////////// Select Point Cloud List File /////////////////////////////////
    select_file();

    printf("\n Point Cloud List File: %s\n", point_cloud_list_file);

    FILE *fp;
    fp = fopen(point_cloud_list_file, "r");
    if (fp != NULL)
    {
        clock_t timer = clock();
        pc_file_open = true;

        char *trash = new char[1024];
        while( fgets(trash,1024,fp) != NULL )
        {
            float3 position;
            uint3 rgb;
            int check = sscanf(trash, "%f %f %f %d %d %d", &position.x, &position.y, &position.z, &rgb.x, &rgb.y, &rgb.z);
            if (check == 6)
            {
                // convert point from mm to meter
                float3 pos_meter = position / 1000.f;
                float3 origin = make_float3( 0.6, 0, 0 );
                float3 vec = pos_meter - origin;
                float distance = length( vec );

                if ( distance < 0.5 ) // check if the point is within x meters of the origin (0,0,0)
                {
                    if (position.x > cloud->max_pos.x) cloud->max_pos.x = position.x;
                    if (position.x < cloud->min_pos.x) cloud->min_pos.x = position.x;

                    if (position.y > cloud->max_pos.y) cloud->max_pos.y = position.y;
                    if (position.y < cloud->min_pos.y) cloud->min_pos.y = position.y;

                    if (position.z > cloud->max_pos.z) cloud->max_pos.z = position.z;
                    if (position.z < cloud->min_pos.z) cloud->min_pos.z = position.z;

                    cloud->position.push_back(position);
                    cloud->rgb.push_back(rgb);
                }
            }
        }
        fclose(fp);

        if ( cloud->position.size() != cloud->rgb.size() )
        {
            printf("\n Error while reading point list. Different number of positions and color channels. Aborting.\n");
        }

        cloud->count = cloud->position.size();
        printf("\n %d points read from file", cloud->count );

        printf("\n Maximum position: %f %f %f", cloud->max_pos.x, cloud->max_pos.y, cloud->max_pos.z );
        printf("\n Minimum position: %f %f %f", cloud->min_pos.x, cloud->min_pos.y, cloud->min_pos.z );

        cloud->data_dim.x = abs( cloud->max_pos.x - cloud->min_pos.x );
        cloud->data_dim.y = abs( cloud->max_pos.y - cloud->min_pos.y );
        cloud->data_dim.z = abs( cloud->max_pos.z - cloud->min_pos.z );
        printf("\n Data Dimensions: %f %f %f", cloud->data_dim.x, cloud->data_dim.y, cloud->data_dim.z );

        cloud->world_origin.x = 0.5 * (cloud->max_pos.x + cloud->min_pos.x);
        cloud->world_origin.y = 0.5 * (cloud->max_pos.y + cloud->min_pos.y);
        cloud->world_origin.z = 0.5 * (cloud->max_pos.z + cloud->min_pos.z);
        printf("\n World Origin: %f %f %f", cloud->world_origin.x, cloud->world_origin.y, cloud->world_origin.z );

        int3 world_size;
        world_size.x = (int)cloud->data_dim.x;
        world_size.y = (int)cloud->data_dim.y;
        world_size.z = (int)cloud->data_dim.z;
        cloud->world_res = 1;
        while ( world_size.x > MAX_VOLUME_SIZE || world_size.y > MAX_VOLUME_SIZE || world_size.z > MAX_VOLUME_SIZE )
        {
            world_size.x /= 2;
            world_size.y /= 2;
            world_size.z /= 2;
            cloud->world_res *= 2;
        }
        cloud->world_size = max( world_size.x, max( world_size.y, world_size.z ) );
        printf("\n World Size: %d", cloud->world_size );

        cloud->world_count = cloud->world_size * cloud->world_size * cloud->world_size;
        printf("\n World Count: %d",cloud->world_count);
        printf("\n World Resolution: %f", cloud->world_res );

        cloud->world_dim = cloud->world_res * (float)cloud->world_size;

        cloud->world_start.x = cloud->world_origin.x - 0.5 * cloud->world_dim;
        cloud->world_start.y = cloud->world_origin.y - 0.5 * cloud->world_dim;
        cloud->world_start.z = cloud->world_origin.z - 0.5 * cloud->world_dim;
        printf("\n World Start: %f %f %f\n", cloud->world_start.x, cloud->world_start.y, cloud->world_start.z );

        printf("\n ||| TIME - Load Data File: %f ms\n", ((float)clock() - timer)*1000 / CLOCKS_PER_SEC );
    }
    else
    {
        printf("\n Could not open file: %s", point_cloud_list_file);
    }
}
void
VR_Window::print_file()
{
    if (pc_file_open)
    {
        printf("\n Current Directory: %s\n", point_cloud_list_file);
    }
}


void
VR_Window::update_render_buffer()
{
    Glib::RefPtr<Gdk::Pixbuf> render_pixbuf = Gdk::Pixbuf::create_from_data((const guint8*) vrender->get_vrender_buffer(),
                                                                                               Gdk::COLORSPACE_RGB,
                                                                                               false,
                                                                                               8,
                                                                                               vrender->get_width(),
                                                                                               vrender->get_height(),
                                                                                               vrender->get_width() * 3 );

    render_image.set( render_pixbuf );
    fps_update.set_text( vrender->get_vrender_fps() );
}
void
VR_Window::set_render_density()
{
    float r_dens = (float) dens_adjust->get_value();
    vrender->set_density(r_dens);
    update_render_buffer();
}
void
VR_Window::set_render_brightness()
{
    float r_bright = (float) bright_adjust->get_value();
    vrender->set_brightness(r_bright);
    update_render_buffer();
}
void
VR_Window::set_render_offset()
{
    float r_offset = (float) offset_adjust->get_value();
    vrender->set_offset(r_offset);
    update_render_buffer();
}
void
VR_Window::set_render_scale()
{
    float r_scale = (float) scale_adjust->get_value();
    vrender->set_scale(r_scale);
    update_render_buffer();
}


void
VR_Window::update_render_zoom(gdouble x, gdouble y)
{
    //printf("\n ZOOM!");
    float dy = (y - vrender->get_last_y()) / 100;

    vrender->set_vrender_zoom( dy );
    update_render_buffer();

    vrender->set_last_x(x);
    vrender->set_last_y(y);
}
void
VR_Window::update_render_translation(gdouble x, gdouble y)
{
    //printf("\n TRANSLATE!");
    float dx = (x - vrender->get_last_x()) / 100;
    float dy = (y - vrender->get_last_y()) / 100;

    vrender->set_vrender_translation( dx, dy );
    update_render_buffer();

    vrender->set_last_x(x);
    vrender->set_last_y(y);
}
void
VR_Window::update_render_rotation(gdouble x, gdouble y)
{
    //printf("\n ROTATE!");
    float dx = (x - vrender->get_last_x()) / 5;
    float dy = (y - vrender->get_last_y()) / 5;

    vrender->set_vrender_rotation( dx, dy );
    update_render_buffer();

    vrender->set_last_x(x);
    vrender->set_last_y(y);
}
bool
VR_Window::render_button_press_event(GdkEventButton *event)
{
    vrender->set_last_x( event->x );
    vrender->set_last_y( event->y );

    /*printf("\n Event-Button: %d\n Button1: %d\n Button2: %d\n Button3: %d\n Condition1: %d\n Condition2: %d\n Condition3: %d\n",
                event->button, GDK_BUTTON_PRIMARY, GDK_BUTTON_SECONDARY, GDK_BUTTON_MIDDLE,
                event->button == GDK_BUTTON_PRIMARY,
                event->button == GDK_BUTTON_SECONDARY,
                event->button == GDK_BUTTON_MIDDLE );*/

    if (event->button == GDK_BUTTON_PRIMARY)
    {
        update_render_rotation(event->x, event->y);
    }
    else if (event->button == GDK_BUTTON_SECONDARY)
    {
        update_render_translation(event->x, event->y);
    }
    else if (event->button == GDK_BUTTON_MIDDLE)
    {
        update_render_zoom(event->x, event->y);
    }
    return true;
}
bool
VR_Window::render_motion_notify_event(GdkEventMotion *event)
{
    /*printf("\n Event-State: %d\n Button1-Mask: %d\n Button2-Mask: %d\n Button3-Mask: %d\n Condition1: %d\n Condition2: %d\n Condition3: %d\n Condition4: %d\n Condition5: %d\n",
                event->state, GDK_BUTTON1_MASK, GDK_BUTTON2_MASK, GDK_BUTTON3_MASK,
                event->state == GDK_BUTTON1_MASK+MASK_BONUS,
                event->state == GDK_BUTTON2_MASK+MASK_BONUS,
                event->state == GDK_BUTTON3_MASK+MASK_BONUS,
                event->state == (GDK_BUTTON1_MASK+GDK_SHIFT_MASK+MASK_BONUS),
                event->state == (GDK_BUTTON1_MASK+GDK_CONTROL_MASK+MASK_BONUS) );*/

    if ( event->state == GDK_BUTTON1_MASK+MASK_BONUS )
    {
        update_render_rotation(event->x, event->y);
    }
    else if ( (event->state == GDK_BUTTON2_MASK+MASK_BONUS) || (event->state == GDK_BUTTON1_MASK+GDK_SHIFT_MASK+MASK_BONUS) )
    {
        update_render_translation(event->x, event->y);
    }
    else if ( (event->state == GDK_BUTTON3_MASK+MASK_BONUS) || (event->state == GDK_BUTTON1_MASK+GDK_CONTROL_MASK+MASK_BONUS) )
    {
        update_render_zoom(event->x, event->y);
    }
    return true;
}


void
VR_Window::create_render_window()
{
    vrender->init_vrender( cloud );

    Gtk::Box *render_vbox = new Gtk::Box(Gtk::ORIENTATION_VERTICAL, 1);

    Gtk::ScrolledWindow *render_scroll;
    Gtk::EventBox       *render_eventbox;

    Glib::RefPtr<Gdk::Pixbuf> render_pixbuf;

    Gtk::Box            *param_hbox1;
    Gtk::Box            *param_hbox2;

    Gtk::Scale          *dens_scale;
    Gtk::Label          *dens_label;

    Gtk::Scale          *bright_scale;
    Gtk::Label          *bright_label;

    Gtk::Scale          *offset_scale;
    Gtk::Label          *offset_label;

    Gtk::Scale          *scale_scale;
    Gtk::Label          *scale_label;

    render_eventbox = new Gtk::EventBox();
    render_eventbox->set_events(  Gdk::BUTTON_PRESS_MASK
                                | Gdk::BUTTON_RELEASE_MASK
                                | Gdk::POINTER_MOTION_MASK
                                | Gdk::POINTER_MOTION_HINT_MASK
                                | Gdk::BUTTON_RELEASE_MASK);

    render_pixbuf->create_from_data((const guint8*) vrender->get_vrender_buffer(),
                                                    Gdk::COLORSPACE_RGB,
                                                    false,
                                                    8,
                                                    vrender->get_width(),
                                                    vrender->get_height(),
                                                    vrender->get_width() * 3 );

    render_image.set( render_pixbuf );
    render_eventbox->add( render_image );

    render_eventbox->signal_motion_notify_event().connect( sigc::mem_fun( *this, &VR_Window::render_motion_notify_event) );
    render_eventbox->signal_button_press_event().connect( sigc::mem_fun( *this, &VR_Window::render_button_press_event) );

    render_scroll = new Gtk::ScrolledWindow();
    render_scroll->set_policy(Gtk::POLICY_AUTOMATIC, Gtk::POLICY_ALWAYS);
    render_scroll->add(render_eventbox[0]);

    fps_label.set_text("Refresh rate (fps):");
    fps_update.set_text("...");

    fps_box.set_orientation( Gtk::ORIENTATION_HORIZONTAL );
    fps_box.set_border_width(0);
    fps_box.pack_start( fps_label, false, false, 2 );
    fps_box.pack_start( fps_update, false, false, 2);
    render_vbox->pack_start( fps_box,  false, false, 2);

    render_vbox->pack_start(render_scroll[0], true, true, 0);

    //////////// CREATE SLIDERS
    param_hbox1 = new Gtk::Box(Gtk::ORIENTATION_HORIZONTAL, 1);
    dens_label = new Gtk::Label("Density: ");
    param_hbox1->pack_start(dens_label[0], false, false, 0);
    dens_adjust = Gtk::Adjustment::create( vrender->get_density(), 0, 1.1, 0.01, 0.1, 0.1);
    dens_adjust->signal_value_changed().connect( sigc::mem_fun( *this, &VR_Window::set_render_density) );
    dens_scale = new Gtk::Scale( dens_adjust, Gtk::ORIENTATION_HORIZONTAL );
    dens_scale->set_digits(2);
    param_hbox1->pack_start(dens_scale[0], true, true, 0);

    bright_label = new Gtk::Label("Brightness: ");
    param_hbox1->pack_start(bright_label[0], false, false, 0);
    bright_adjust = Gtk::Adjustment::create( vrender->get_brightness(), 0, 10.1, 0.1, 0.1, 0.1);
    bright_adjust->signal_value_changed().connect( sigc::mem_fun( *this, &VR_Window::set_render_brightness) );
    bright_scale = new Gtk::Scale( bright_adjust, Gtk::ORIENTATION_HORIZONTAL );
    bright_scale->set_digits(2);
    param_hbox1->pack_start(bright_scale[0], true, true, 0);
    render_vbox->pack_start(param_hbox1[0], false, false, 0);

    param_hbox2 = new Gtk::Box(Gtk::ORIENTATION_HORIZONTAL, 1);
    offset_label = new Gtk::Label("Offset: ");
    param_hbox2->pack_start(offset_label[0], false, false, 0);
    offset_adjust = Gtk::Adjustment::create( vrender->get_offset(), -1, 1.1, 0.01, 0.1, 0.1);
    offset_adjust->signal_value_changed().connect( sigc::mem_fun( *this, &VR_Window::set_render_offset) );
    offset_scale = new Gtk::Scale( offset_adjust, Gtk::ORIENTATION_HORIZONTAL );
    offset_scale->set_digits(2);
    param_hbox2->pack_start(offset_scale[0], true, true, 0);

    scale_label = new Gtk::Label("Scale: ");
    param_hbox2->pack_start(scale_label[0], false, false, 0);
    scale_adjust = Gtk::Adjustment::create( vrender->get_scale(), 0, 10.5, 0.1, 0.5, 0.5);
    scale_adjust->signal_value_changed().connect( sigc::mem_fun( *this, &VR_Window::set_render_scale) );
    scale_scale = new Gtk::Scale( scale_adjust, Gtk::ORIENTATION_HORIZONTAL );
    scale_scale->set_digits(2);
    param_hbox2->pack_start(scale_scale[0], true, true, 0);
    render_vbox->pack_start(param_hbox2[0], false, false, 0);

    pack_start(render_vbox[0], true, true, 0);
    show_all_children();

    update_render_buffer();
}

