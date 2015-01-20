/***********************************************************
     Starter code for Assignment 3

     This code was originally written by Jack Wang for
		    CSC418, SPRING 2005

		Implementations of functions in raytracer.h, 
		and the main function which specifies the 
		scene to be rendered.	

***********************************************************/


#include "raytracer.h"
#include "bmp_io.h"
#include <cmath>
#include <iostream>
#include <cstdlib>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/wait.h>

Raytracer::Raytracer() : _lightSource(NULL) {
	_root = new SceneDagNode();
}

Raytracer::~Raytracer() {
	delete _root;
}

SceneDagNode* Raytracer::addObject( SceneDagNode* parent, 
		SceneObject* obj, Material* mat ) {
	SceneDagNode* node = new SceneDagNode( obj, mat );
	node->parent = parent;
	node->next = NULL;
	node->child = NULL;
	
	// Add the object to the parent's child list, this means
	// whatever transformation applied to the parent will also
	// be applied to the child.
	if (parent->child == NULL) {
		parent->child = node;
	}
	else {
		parent = parent->child;
		while (parent->next != NULL) {
			parent = parent->next;
		}
		parent->next = node;
	}
	
	return node;;
}


SceneDagNode* Raytracer::addBox(Material* mat){

    SceneDagNode* node = addObject(_root,NULL,mat);
    SceneDagNode* s[6];
    
    for(int i=0;i<6;i++)
        s[i] = addObject(node,new UnitSquare(), mat);
    
    
    translate(s[0], Vector3D(0, 0.5, 0));	
	rotate(s[0], 'x', -90); 

    translate(s[1], Vector3D(0, -0.5, 0));	
	rotate(s[1], 'x', 90); 

    translate(s[2], Vector3D(-0.5, 0, 0));	
	rotate(s[2], 'y', -90); 

    translate(s[3], Vector3D(0.5, 0, 0));	
	rotate(s[3], 'y', 90); 

    translate(s[4], Vector3D(0, 0, 0.5));	

    translate(s[5], Vector3D(0, 0, -0.5));
	rotate(s[5], 'y', 180); 

    return node;
    
}

LightListNode* Raytracer::addLightSource( LightSource* light ) {
	LightListNode* tmp = _lightSource;
	_lightSource = new LightListNode( light, tmp );
	return _lightSource;
}

void Raytracer::rotate( SceneDagNode* node, char axis, double angle ) {
	Matrix4x4 rotation;
	double toRadian = 2*M_PI/360.0;
	int i;
	
	for (i = 0; i < 2; i++) {
		switch(axis) {
			case 'x':
				rotation[0][0] = 1;
				rotation[1][1] = cos(angle*toRadian);
				rotation[1][2] = -sin(angle*toRadian);
				rotation[2][1] = sin(angle*toRadian);
				rotation[2][2] = cos(angle*toRadian);
				rotation[3][3] = 1;
			break;
			case 'y':
				rotation[0][0] = cos(angle*toRadian);
				rotation[0][2] = sin(angle*toRadian);
				rotation[1][1] = 1;
				rotation[2][0] = -sin(angle*toRadian);
				rotation[2][2] = cos(angle*toRadian);
				rotation[3][3] = 1;
			break;
			case 'z':
				rotation[0][0] = cos(angle*toRadian);
				rotation[0][1] = -sin(angle*toRadian);
				rotation[1][0] = sin(angle*toRadian);
				rotation[1][1] = cos(angle*toRadian);
				rotation[2][2] = 1;
				rotation[3][3] = 1;
			break;
		}
		if (i == 0) {
		    node->trans = node->trans*rotation; 	
			angle = -angle;
		} 
		else {
			node->invtrans = rotation*node->invtrans; 
		}	
	}
}

void Raytracer::translate( SceneDagNode* node, Vector3D trans ) {
	Matrix4x4 translation;
	
	translation[0][3] = trans[0];
	translation[1][3] = trans[1];
	translation[2][3] = trans[2];
	node->trans = node->trans*translation; 	
	translation[0][3] = -trans[0];
	translation[1][3] = -trans[1];
	translation[2][3] = -trans[2];
	node->invtrans = translation*node->invtrans; 
}

void Raytracer::scale( SceneDagNode* node, Point3D origin, double factor[3] ) {
	Matrix4x4 scale;
	
	scale[0][0] = factor[0];
	scale[0][3] = origin[0] - factor[0] * origin[0];
	scale[1][1] = factor[1];
	scale[1][3] = origin[1] - factor[1] * origin[1];
	scale[2][2] = factor[2];
	scale[2][3] = origin[2] - factor[2] * origin[2];
	node->trans = node->trans*scale; 	
	scale[0][0] = 1/factor[0];
	scale[0][3] = origin[0] - 1/factor[0] * origin[0];
	scale[1][1] = 1/factor[1];
	scale[1][3] = origin[1] - 1/factor[1] * origin[1];
	scale[2][2] = 1/factor[2];
	scale[2][3] = origin[2] - 1/factor[2] * origin[2];
	node->invtrans = scale*node->invtrans; 
}

Matrix4x4 Raytracer::initInvViewMatrix( Point3D eye, Vector3D view, 
		Vector3D up ) {
	Matrix4x4 mat; 
	Vector3D w;
	view.normalize();
	up = up - up.dot(view)*view;
	up.normalize();
	w = view.cross(up);

	mat[0][0] = w[0];
	mat[1][0] = w[1];
	mat[2][0] = w[2];
	mat[0][1] = up[0];
	mat[1][1] = up[1];
	mat[2][1] = up[2];
	mat[0][2] = -view[0];
	mat[1][2] = -view[1];
	mat[2][2] = -view[2];
	mat[0][3] = eye[0];
	mat[1][3] = eye[1];
	mat[2][3] = eye[2];

	return mat; 
}

void Raytracer::traverseScene( SceneDagNode* node, Ray3D& ray ) {
	SceneDagNode *childPtr;

	// Applies transformation of the current node to the global
	// transformation matrices.
	_modelToWorld = _modelToWorld*node->trans;
	_worldToModel = node->invtrans*_worldToModel; 
	if (node->obj) {
		// Perform intersection.
		if (node->obj->intersect(ray, _worldToModel, _modelToWorld)) {
			ray.intersection.mat = node->mat;
		}
	}
	// Traverse the children.
	childPtr = node->child;
	while (childPtr != NULL) {
		traverseScene(childPtr, ray);
		childPtr = childPtr->next;
	}

	// Removes transformation of the current node from the global
	// transformation matrices.
	_worldToModel = node->trans*_worldToModel;
	_modelToWorld = _modelToWorld*node->invtrans;
}

void Raytracer::computeShading( Ray3D& ray, int depth ) {
	LightListNode* curLight = _lightSource;
    
    Colour col(0,0,0);
    double tl,t2;
    Ray3D lray;	
    Vector3D ldir,sdir;
    double offset = 0.1;
    Vector3D n(ray.intersection.normal);


    for (;;) {
		if (curLight == NULL) break;
		// Each lightSource provides its own shading function.

        // Add Shadow/Local shading
        col = col + traceShadow(curLight->light, ray, depth);
       
        // Add Refraction/Reflection 
        col = col + traceRefraction( ray, depth);
        
        // Add Ambient color
        ray.col = Colour(0,0,0);
        curLight->light->shadeAmbient(ray);
        col = col + ray.col;

        // Clamp
        col.clamp();
         
		curLight = curLight->next;
        
	}

    ray.col = col;
}

Colour Raytracer::traceShadow(LightSource *light, Ray3D& ray, int depth){

    int samples;
    double x,y;
    Point3D light_pos;
    Colour col(0,0,0);            
    double offset = 0.01;

    // Clear Ray's color
    ray.col = Colour(0,0,0);

    if(light->get_type() == LightSource::POINT)
        samples = 1;
    else
        samples = _shadow_samples;

    for(int i=0;i<samples;i++){
        for(int j=0;j<samples;j++){
            Ray3D sray;
            Vector3D sdir;
            double t;

            if(samples == 1){
                x = 0.0;
                y = 0.0;
            } else {
                //x = i/double(samples) + rand()/(double(RAND_MAX) * samples);
                x = (1/double(samples) * 
                        (i + rand()/double(RAND_MAX))) - 0.5;
                y = (1/double(samples) * 
                        (j + rand()/double(RAND_MAX))) - 0.5;

                //y = j/double(samples) + rand()/(double(RAND_MAX) * samples);
            }

            light->set_offset(x,y);

            sdir = light->get_position() - 
                ray.intersection.point;

            sray.dir    = sdir;
            sray.dir.normalize();
            sray.origin = ray.intersection.point + (offset * sray.dir);
            sray.intersection.none = true;

            if(sray.dir.dot(ray.intersection.normal) < 0)
                continue;

            // caution: divide by zero is possible
            t = sdir[0] /sray.dir[0];    

            traverseScene(_root, sray); 
            if(sray.intersection.none || t < sray.intersection.t_value){
                if(ray.intersection.mat->type == DIFFUSE)
                    light->shade(ray);
            }
        }
    }

    ray.col = (1.00/(samples*samples)) * ray.col;
    ray.col.clamp();

    return ray.col;

}

Colour Raytracer::traceReflection(Ray3D& ray, int depth){

    // ********** Specular/Mirror Reflection
    Vector3D n(ray.intersection.normal);
    Vector3D w;
    Vector3D u,v,up;
    int gloss_samples = _gloss_samples;
    double gloss_factor = ray.intersection.mat->gloss;
    double offset = 0.01;
   
    Ray3D mray;
    mray.dir    = ray.dir - ((2 * (ray.dir.dot(n)))*n);
    mray.dir.normalize(); 
    mray.origin = ray.intersection.point + (offset * mray.dir);
    
    // Clear ray's colour
    ray.col = Colour(0,0,0);

    // If glossiness is 0.00, don't bother mutlisampling
    if(gloss_factor <= 0) 
        gloss_samples = 1;


    // If the perfect reflection ray is above surface and 
    // material is dielectric
    if(mray.dir.dot(ray.intersection.normal) > 0 && 
            ray.intersection.mat->type == DIELECTRIC){

        if(mray.dir[0] > mray.dir[1] && mray.dir[0] > mray.dir[2])
            up = Vector3D(1,0,0);
        else if(mray.dir[1] > mray.dir[0] && mray.dir[1] > mray.dir[2])
            up = Vector3D(0,1,0);
        else
            up = Vector3D(0,0,1);

        w = (mray.dir);
        u = w.cross(up);
        v = up.cross(u);
        u.normalize();
        v.normalize();

        for(int i=0;i<gloss_samples;i++){
            for(int j=0;j<gloss_samples;j++){
                Ray3D rray;
                double x,y;

                x = ((rand()/double(RAND_MAX)) - 0.5);
                y = ((rand()/double(RAND_MAX)) - 0.5);
                //x = (i/double(gloss_samples) + 
                //    (rand()/(double(RAND_MAX)*gloss_samples)))  - 0.5;
                //y = (j/double(gloss_samples) +
                //    (rand()/(double(RAND_MAX)*gloss_samples)))  - 0.5;



                x *= gloss_factor;
                y *= gloss_factor;

                rray.dir = mray.dir + (x * u) + (y * v);
                rray.dir.normalize();
                rray.origin = ray.intersection.point + (offset * rray.dir);

                // Ray above surface
                if(ray.intersection.normal.dot(rray.dir) > 0)
                    ray.col = ray.col + 
                        (1.00/(gloss_samples*gloss_samples)) *
                        //(ray.intersection.mat->specular *  
                        (shadeRay(rray, depth+1));
            }
        } 
        //ray.col = (1.00/gloss_samples*gloss_samples) * ray.col;
        ray.col.clamp();

    }

    return ray.col;
}

Colour Raytracer::traceRefraction(Ray3D& ray, int depth){


    // ****** Refraction
    Ray3D tray;
    Vector3D d  = ray.dir;
    Vector3D n  = ray.intersection.normal;
    double t  = ray.intersection.t_value;
    double dn = ray.dir.dot(ray.intersection.normal);
    double offset = 0.01;
    double nt = ray.intersection.mat->exp;  // IOR of Material
    double c;
    double r0,r1;      // Schlick's approx to Fresnel equations
    Colour k(0,0,0); // k(r,g,b) in Beer's law
    Colour a = ray.intersection.mat->specular; // Attenuation coefficient        
    ray.col = Colour(0,0,0); // Clear ray's Colour
    Colour col;

    // This is bit of a hack to speed up up perfect reflection
    // if IOR is big enough, don't bother with refraction
    //if(nt > 100)
    //    return traceReflection(ray,depth+1);


    if(ray.intersection.mat->type == DIELECTRIC){
        // Going into object
        if(dn < 0){
            double nr = 1.00/nt;
            double root = (1 - (nr*nr* (1-(dn*dn))));
            c = -1 * dn;
            k = Colour(1,1,1);            
    
            // Sanity check
            if(root >= 0){
                tray.dir = (nr * (d - (dn * n))) - (sqrt(root) * n);
                tray.dir.normalize();
                tray.origin = ray.intersection.point + (offset * tray.dir);
            }
            else {
                // Should not be reached
                std::cout << "traceRefraction: root is negative.\n";
                return Colour(0,0,0);
            }

        }
        // Going out of object 
        else {
            double nr = nt;
            k[0] = exp(-1 * a[0] * t);     
            k[1] = exp(-1 * a[1] * t);     
            k[2] = exp(-1 * a[2] * t);     

            n  = -1 * n;
            dn = ray.dir.dot(n);;            

            double root = (1 - (nr*nr* (1-(dn*dn))));

            if(root >= 0){
                // No total internal reflection
                tray.dir = (nr * (d - (dn * n))) - (sqrt(root) * n);
                tray.dir.normalize();
                tray.origin = ray.intersection.point + (offset * tray.dir);
                c = tray.dir.dot(-1 * n);
            }
            else {
                // Total internal reflection
                return k * traceReflection(ray,depth+1);
            }
        }

        // Schlick's approximation
        r0 = ((nt-1)*(nt-1))/((nt+1)*(nt+1));
        r1 = r0 + ((1-r0) * pow(1-c,5));

        col = r1 * traceReflection(ray, depth+1);       // Reflection             
        col = col + (1 - r1)  * shadeRay(tray,depth+1); // Refraction
        col = k * col;                                  // Beer's law

    }

    return col;

}

void Raytracer::initPixelBuffer() {
    int numbytes = _scrWidth * _scrHeight * sizeof(unsigned char);
    //_rbuffer = new unsigned char[numbytes];
    //_gbuffer = new unsigned char[numbytes];
    //_bbuffer = new unsigned char[numbytes];
    _rbuffer = (unsigned char*) mmap(NULL, numbytes, PROT_READ|PROT_WRITE, 
            MAP_ANON|MAP_SHARED, -1, 0);
    _gbuffer = (unsigned char*) mmap(NULL, numbytes, PROT_READ|PROT_WRITE, 
            MAP_ANON|MAP_SHARED, -1, 0);
    _bbuffer = (unsigned char*) mmap(NULL, numbytes, PROT_READ|PROT_WRITE, 
            MAP_ANON|MAP_SHARED, -1, 0);

    for (int i = 0; i < _scrHeight; i++) {
        for (int j = 0; j < _scrWidth; j++) {
            _rbuffer[i*_scrWidth+j] = 0;
            _gbuffer[i*_scrWidth+j] = 0;
            _bbuffer[i*_scrWidth+j] = 0;
        }
    }
}

void Raytracer::flushPixelBuffer( char *file_name ) {
    bmp_write( file_name, _scrWidth, _scrHeight, _rbuffer, _gbuffer, _bbuffer );
    int numbytes = _scrWidth * _scrHeight * sizeof(unsigned char);
    munmap(_rbuffer,numbytes);
    munmap(_gbuffer,numbytes);
    munmap(_bbuffer,numbytes);
    /*
       delete _rbuffer;
       delete _gbuffer;
       delete _bbuffer;
       */
}

Colour Raytracer::shadeRay( Ray3D& ray, int depth) {
    Colour col(0.0, 0.0, 0.0); 
    traverseScene(_root, ray); 
	
	// Don't bother shading if the ray didn't hit 
	// anything or reached maximum depth.
	if (!ray.intersection.none && depth < _max_depth) {
		computeShading(ray,depth); 
		col = ray.col;  
		
        // Scene Signature
        //col = ray.intersection.mat->diffuse;
        
        // Normal Map
        Vector3D nmap(ray.intersection.normal);
        nmap = Vector3D(0.5,0.5,0.5) + (0.5 * nmap);
        
        //col = Colour(nmap[0],nmap[1],nmap[2]);
	}

	// You'll want to call shadeRay recursively (with a different ray, 
	// of course) here to implement reflection/refraction effects.  

	return col; 
}	

void Raytracer::render( int width, int height, Point3D eye, Vector3D view, 
		Vector3D up, double fov, char* fileName ) {
	Matrix4x4 viewToWorld;
	_scrWidth = width;
	_scrHeight = height;
    
    double factor = (double(height)/2)/tan(fov*M_PI/360.0);

	initPixelBuffer();
	viewToWorld = initInvViewMatrix(eye, view, up);

    // Create _max_threads processes
    // each of which is responsible
    // 1/_max_threads rows of pixels
    process_data pd[_max_threads];
    pid_t   pid[_max_threads];
    int     status[_max_threads];

    for(int i=0;i<_max_threads;i++){
        pd[i].pnum = i;
        pd[i].factor = factor;
        pd[i].viewToWorld = viewToWorld;
        pd[i].pid = fork();
        if(pd[i].pid < 0){
            // Error
            std::cout << "Failed to create process " << i << "\n";
            exit(0);
        }        
        else if(pd[i].pid == 0){
            // Child process
            render_thread(&pd[i]);
            exit(0);
        }
    }
   
    for(int i=0;i<_max_threads;i++){
        while( waitpid(pd[i].pid,&status[i],0) == -1 );
        std::cout << "\nProcess " << i << " finished.";
    }

    std::cout << "\nCompleted.\n" ;

    flushPixelBuffer(fileName);
}

void Raytracer::render_thread(process_data *pd){
   
    // Local variables
    int         pid    = pd->pid;
    int         pnum   = pd->pnum;
    double      factor = pd->factor;   
    Matrix4x4   viewToWorld = pd->viewToWorld;
    int         width  = _scrWidth;
    int         height = _scrHeight;
    int         size   = height / _max_threads;
    int         start  = pnum * size;
    int         end    = (pnum+1)*size;
    int         ft     = 10;      // Focal distance
    int         dof_samples = _dof_samples; // Depth of field samples
    Colour      col(0,0,0);
    
    // Construct a ray for each pixel at rows [_start, _end)
	for (int i = start; i < end; i++) {
        std::cout.precision(2);
        if(pnum == 0){
            std::cout << std::fixed << "\r"
                      << (i - start)/double(end - start) << "%       " 
                      << std::flush;
        }
		for (int j = 0; j < width; j++) {
			
            // Clear colour buffer
            col = Colour(0.0,0.0,0.0);
            
            // Construct aa_samples rays per pixel
            for(int rx = 0; rx < _aa_samples;rx++) {
                for(int ry = 0; ry < _aa_samples;ry++){
                
                // Sets up ray origin and direction in view space, 
			    // image plane is at z = -1.
			    Point3D origin(0, 0, 0);
			    Point3D imagePlane;
          
                 
                //rx = (double)rand()/RAND_MAX;
                //ry = (double)rand()/RAND_MAX;
                
                if(_aa_samples == 1){
                    rx = 0.5;
                    ry = 0.5;
                }
            

			    imagePlane[0] = (-double(width)/2 + 
                                (rx+0.5)/(double)_aa_samples + j)/factor;
			    imagePlane[1] = (-double(height)/2 + 
                                (ry+0.5)/(double)_aa_samples + i)/factor;
			    imagePlane[2] = -1;

                // Depth of Field
                // Find focal point
                // For dof_samples:
                //    Find new origin
                //    Converd origin and focal point to WorldCoords
                //    Ray = focal point - new origin
                //    col = col + (1/dof_samples) * shadeRay(new origin)
                //
                 
                Vector3D u(3,0,0); // set aperture here
                Vector3D v(0,3,0);
                Vector3D rayDirection = imagePlane - origin;        
                rayDirection.normalize();            
                Point3D focalPoint = origin + (ft * rayDirection);
                // Convert to world space
                //focalPoint = viewToWorld * focalPoint;

                for(int fi=0;fi<dof_samples;fi++) 
                 for(int fj=0;fj<dof_samples;fj++){
                    
                    Ray3D ray;
                    //double ru = (fi/double(dof_samples) * 
                    //            (rand()/double(RAND_MAX))) - 0.5;
                    //double rv = (fj/double(dof_samples) * 
                    //            (rand()/double(RAND_MAX))) - 0.5;
                    //double ru = fi/double(dof_samples) - 0.5;
                    //double rv = fj/double(dof_samples) - 0.5;
                    //ru += rand()/double(RAND_MAX * dof_samples);
                    //rv += rand()/double(RAND_MAX * dof_samples);
                    double ru = rand()/double(RAND_MAX) - 0.5;
                    double rv = rand()/double(RAND_MAX) - 0.5;

                    if(dof_samples == 1){
                        ru = 0;
                        rv = 0;
                    }
    
                    Point3D newOrigin = origin + (ru * u) + (rv * v);
                    ray.dir = viewToWorld * (focalPoint - newOrigin);
                    ray.dir.normalize();
                    ray.origin = viewToWorld * newOrigin;
                    col = col + shadeRay(ray);
                    
                }

                col = (1.00/double(dof_samples*dof_samples)) * col;
                
                /* 
			    // TODO: Convert ray to world space and call
                origin     = viewToWorld * origin;
                imagePlane = viewToWorld * imagePlane;
                     


			    // shadeRay(ray) to generate pixel colour. 	
			
			    Ray3D ray;
                ray.origin = origin;
                ray.dir    = imagePlane - origin;        
                ray.dir.normalize();            
	    		col = col + shadeRay(ray); 
                */
                }
            }

            col[0] = col[0]/double(_aa_samples*_aa_samples);
            col[1] = col[1]/double(_aa_samples*_aa_samples);
            col[2] = col[2]/double(_aa_samples*_aa_samples);

            col.clamp();
            
           
            _rbuffer[i*width+j] = int(col[0]*255);
            _gbuffer[i*width+j] = int(col[1]*255);
            _bbuffer[i*width+j] = int(col[2]*255);
        }
	}

}


int main(int argc, char* argv[])
{	
	// Build your scene and setup your camera here, by calling 
	// functions from Raytracer.  The code here sets up an example
	// scene and renders it from two different view points, DO NOT
	// change this if you're just implementing part one of the 
	// assignment.  
	Raytracer raytracer;

    srand(time(NULL));

	int width = 320; 
	int height = 240; 

	if (argc == 3) {
		width = atoi(argv[1]);
		height = atoi(argv[2]);
	}

	// Camera parameters.
	Point3D eye(0, 0, 7);
	Vector3D view(0, 0, -1);
	Vector3D up(0, 1, 0);
	double fov = 60;

	// Defines a material for shading.
	
	Material gold(DIELECTRIC, Colour(0, 0, 0), Colour(0.75164, 0.60648, 0.22648), 
			Colour(0.628281, 0.555802, 0.366065), 
			200, 0.0);
	
    Material jade(DIFFUSE, Colour(0.1, 0.1, 0.1), Colour(0.54, 0.89, 0.63), 
			Colour(0.316228, 0.316228, 0.316228), 
			12.8, 0.0);
    

	Material gray(DIFFUSE, Colour(0.1, 0.1,0.1), Colour(0.7, 0.7, 0.7), 
			Colour(0.316228, 0.316228, 0.316228), 
			12.8, 0.0);
	
	Material red(DIFFUSE, Colour(0.1, 0.1, 0.1), Colour(0.5, 0.1, 0.1), 
			Colour(0.316228, 0.316228, 0.316228), 
			12.8, 0.0);
	Material green(DIFFUSE, Colour(0.1, 0.1, 0.1), Colour(0.1, 0.5, 0.1), 
			Colour(0.316228, 0.316228, 0.316228), 
			12.8, 0.0);
    
	Material blue(DIFFUSE, Colour(0.1, 0.1, 0.1), Colour(0.6, 0.8, 0.9), 
			Colour(0.316228, 0.316228, 0.316228), 
			12.8, 0.0);
    
	Material chrome(DIELECTRIC,  Colour(0, 0, 0), Colour(0.0, 0.0, 0.0), 
			//Colour(0.916228, 0.916228, 0.916228), 
			Colour(0.0, 0.0, 0.0), 
			300, 0.0);

	Material glossy_chrome(DIELECTRIC,  Colour(0, 0, 0), Colour(0.0, 0.0, 0.0), 
			//Colour(0.916228, 0.916228, 0.916228), 
			Colour(0.0, 0.0, 0.0), 
			300, 0.9);
	
    Material glass(DIELECTRIC, Colour(0, 0, 0), Colour(0.0, 0.0, 0.0), 
			Colour(0.0, 0.0, 0.0), 
			1.6, 0.0);

    Material green_glass(DIELECTRIC, Colour(0, 0, 0), Colour(0.0, 0.0, 0.0), 
			Colour(0.1, 0.0, 0.1), 
			1.6, 0.0);
	
    Material white(DIFFUSE, Colour(0.9, 0.9, 0.9), Colour(1.0, 1.0, 1.0), 
			Colour(0.9, 0.9, 0.9), 
			30, 0.0 );
    
    
    // Defines a point light source.
	//raytracer.addLightSource( new PointLight(Point3D(2,4,-5), 
	//          Colour(0.9, 0.9, 0.9) ) );

	raytracer.addLightSource( new AreaLight(Point3D(0,3.9,-5), 
	          Vector3D(3,0,0), Vector3D(0,0,3) ,Colour(0.9, 0.9, 0.9) ) );
	
    // Add a unit square into the scene with material mat.
	SceneDagNode* sphere = raytracer.addObject( new UnitSphere(), &chrome);
	SceneDagNode* sphere2 = raytracer.addObject( new UnitSphere(), &glass);
	//SceneDagNode* plane2 = raytracer.addObject( new UnitSquare(), &gold );
	
    
    SceneDagNode* ceil = raytracer.addObject( new UnitSquare(), &gray );
    SceneDagNode* floor = raytracer.addObject( new UnitSquare(), &gray );
    SceneDagNode* left = raytracer.addObject( new UnitSquare(), &red );
    SceneDagNode* right = raytracer.addObject( new UnitSquare(), &green );
    SceneDagNode* back = raytracer.addObject( new UnitSquare(), &gray );
    
    SceneDagNode* box = raytracer.addBox(&jade);
    SceneDagNode* box2 = raytracer.addBox(&chrome);
    SceneDagNode* lightbox = raytracer.addBox(&white);

	
	// Apply some transformations to the unit square.
	double factor1[3] = { 1.5, 1.5, 1.5 };
	double factor2[3] = { 5.0, 5.0, 0.6 };
	double factor3[3] = { 10.0, 10.0, 10.0 };
	double factor4[3] = { 3.0, 1.0 , 3.0 };
	double factor5[3] = { 2.0, 2.0 , 2.0 };
	double factor6[3] = { 3.0, 3.0 , 0.5 };
    
    

	
    raytracer.translate(sphere, Vector3D(2.5, -3.5, -7));	
	raytracer.scale(sphere, Point3D(0, 0, 0), factor1);
	
    raytracer.translate(sphere2, Vector3D(0, 0, -2));	
	raytracer.scale(sphere2, Point3D(0, 0, 0), factor6);
    
    raytracer.translate(floor, Vector3D(0, -5.0, -5));	
	raytracer.rotate(floor, 'x', -90); 
	raytracer.scale(floor, Point3D(0, 0, 0), factor3);

    raytracer.translate(ceil, Vector3D(0, 5.0, -5));	
	raytracer.rotate(ceil, 'x', 90); 
	raytracer.scale(ceil, Point3D(0, 0, 0), factor3);
    
    raytracer.translate(left, Vector3D(-5.0, 0, -5));	
	raytracer.rotate(left, 'y', 90); 
	raytracer.scale(left, Point3D(0, 0, 0), factor3);
    
    raytracer.translate(right, Vector3D(5.0, 0, -5));	
	raytracer.rotate(right, 'y', -90); 
	raytracer.scale(right, Point3D(0, 0, 0), factor3);
    
    raytracer.translate(back, Vector3D(0, 0, -10));	
	raytracer.scale(back, Point3D(0, 0, 0), factor3);


    raytracer.translate(box, Vector3D(-2, -4, -5));	
	//raytracer.rotate(box, 'x', -45);
	raytracer.rotate(box, 'y', -45);
	raytracer.scale(box, Point3D(0, 0, 0), factor5);
    
    raytracer.translate(box2, Vector3D(-2, -2.5, -60));	
	raytracer.rotate(box2, 'y', 45);
	raytracer.scale(box2, Point3D(0, 0, 0), factor2);
    
    raytracer.translate(lightbox, Vector3D(0, 5.2, -5));	
	raytracer.scale(lightbox, Point3D(0, 0, 0), factor4);
    

	// Render the scene, feel free to make the image smaller for
	// testing purposes.	
	raytracer.render(width, height, eye, view, up, fov, "view1.bmp");
	
	// Render it from a different point of view.
	Point3D eye2(4, 2, 1);
	Vector3D view2(-4, -2, -6);
	raytracer.render(width, height, eye2, view2, up, fov, "view2.bmp");
	
	return 0;
}

