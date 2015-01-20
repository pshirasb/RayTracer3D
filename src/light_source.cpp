/***********************************************************
     Starter code for Assignment 3

     This code was originally written by Jack Wang for
		    CSC418, SPRING 2005

		implements light_source.h

***********************************************************/

#include <cmath>
#include "light_source.h"

void PointLight::shade( Ray3D& ray ) {
	// TODO: implement this function to fill in values for ray.col 
	// using phong shading.  Make sure your vectors are normalized, and
	// clamp colour values to 1.0.
	//
	// It is assumed at this point that the intersection information in ray 
	// is available.  So be sure that traverseScene() is called on the ray 
	// before this function.  
    
    double sn,nh;    

    // calculate max(0,(s.n))
    // s = light direction
    // n = normal
    Vector3D s((_pos - ray.intersection.point));
    Vector3D n(ray.intersection.normal);
    s.normalize();
    n.normalize();
    sn = s.dot(n);
    if(sn < 0)
        sn = 0;
    
    // calculate max(0,(n.h))    
    Vector3D v(-ray.dir);
    v.normalize();
    Vector3D h(v+s);
    h.normalize(); 
    nh = n.dot(h);
    if(nh < 0)
        nh = 0;
    
    // Apply specular exponent
    nh = pow(nh,ray.intersection.mat->exp);    
    
    // Phong equation
    ray.col = ray.col + //_col_ambient * ray.intersection.mat->ambient +
              //(1 - ray.intersection.mat->trans) *  
              sn * _col_diffuse  * ray.intersection.mat->diffuse + 
              nh * _col_specular * ray.intersection.mat->specular;

    //ray.col.clamp(); 

}

void PointLight::shadeAmbient( Ray3D& ray ){
    ray.col  = ray.col + _col_ambient * ray.intersection.mat->ambient;
}

void AreaLight::shade( Ray3D& ray ) {
	// TODO: implement this function to fill in values for ray.col 
	// using phong shading.  Make sure your vectors are normalized, and
	// clamp colour values to 1.0.
	//
	// It is assumed at this point that the intersection information in ray 
	// is available.  So be sure that traverseScene() is called on the ray 
	// before this function.  
    
    double sn,nh;    

    // calculate max(0,(s.n))
    // s = light direction
    // n = normal
    Vector3D s(get_position() - ray.intersection.point);
    Vector3D n(ray.intersection.normal);
    s.normalize();
    n.normalize();
    sn = s.dot(n);
    if(sn < 0)
        sn = 0;
    
    // calculate max(0,(n.h))    
    Vector3D v(-ray.dir);
    v.normalize();
    Vector3D h(v+s);
    h.normalize(); 
    nh = n.dot(h);
    if(nh < 0)
        nh = 0;
    
    // Apply specular exponent
    nh = pow(nh,ray.intersection.mat->exp);    
    
    // Phong equation
    ray.col = ray.col + //_col_ambient * ray.intersection.mat->ambient +
              sn * _col_diffuse  * ray.intersection.mat->diffuse + 
              nh * _col_specular * ray.intersection.mat->specular;

    //ray.col.clamp(); 

}

void AreaLight::shadeAmbient( Ray3D& ray ){
    ray.col  = ray.col + _col_ambient * ray.intersection.mat->ambient;
    //ray.col.clamp();
}
