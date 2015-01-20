/***********************************************************
     Starter code for Assignment 3

     This code was originally written by Jack Wang for
		    CSC418, SPRING 2005

		   light source classes

***********************************************************/

#include "util.h"

// Base class for a light source.  You could define different types
// of lights here, but point light is sufficient for most scenes you
// might want to render.  Different light sources shade the ray 
// differently.
class LightSource {
public:

    enum LightType {POINT, AREA};
  
	virtual void shade( Ray3D& ) = 0;
    virtual void shadeAmbient( Ray3D& ) = 0;
	virtual Point3D get_position() const = 0; 
    virtual LightType get_type() const = 0;
    virtual void set_offset( double, double ) = 0;
    
};

// A point light is defined by its position in world space and its
// colour.
class PointLight : public LightSource {
public:
	PointLight( Point3D pos, Colour col ) : _pos(pos), _col_ambient(col), 
	_col_diffuse(col), _col_specular(col), _type(POINT) {}
	PointLight( Point3D pos, Colour ambient, Colour diffuse, Colour specular ) 
	: _pos(pos), _col_ambient(ambient), _col_diffuse(diffuse), 
	_col_specular(specular), _type(POINT) {}
	void shade( Ray3D& ray );
    void shadeAmbient( Ray3D& ray );
	Point3D get_position() const { return _pos; }
    LightType get_type() const { return _type; }
    void set_offset(double, double) { return; }
	
private:
	Point3D _pos;
	Colour _col_ambient;
	Colour _col_diffuse; 
	Colour _col_specular; 
    LightType   _type;
};

// An area light is defined by a corner point and
// two orthogonal side vectors in world space.
// colour.
class AreaLight : public LightSource {
public:
	AreaLight( Point3D pos, Colour col ) : _pos(pos), _col_ambient(col), 
	_col_diffuse(col), _col_specular(col), _type(AREA) {}
	
    AreaLight( Point3D pos, Vector3D u, Vector3D v, Colour col ) 
    : _pos(pos), _col_ambient(col), _u(u), _v(v), _x(0), _y(0),
	_col_diffuse(col), _col_specular(col), _type(AREA) {}
	
    AreaLight( Point3D pos, Colour ambient, Colour diffuse, Colour specular ) 
	: _pos(pos), _col_ambient(ambient), _col_diffuse(diffuse), 
	_col_specular(specular), _type(AREA) {}
    

	void shade( Ray3D& ray );
    void shadeAmbient( Ray3D& ray );
	Point3D   get_position() const { 
        return _pos + (_x * _u) + (_y * _v); 
    }
    LightType get_type() const { return _type; }
    Vector3D  get_u() { return _u; }
    Vector3D  get_v() { return _v; }
    void      set_offset(double x, double y) { _x = x; _y = y; } 
    
	
private:
	Point3D  _pos;
	Vector3D _u;
    Vector3D _v; 
    Colour _col_ambient;
	Colour _col_diffuse; 
	Colour _col_specular;
    LightType   _type;
    double _x;
    double _y; 
};
