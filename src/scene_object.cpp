/***********************************************************
     Starter code for Assignment 3

     This code was originally written by Jack Wang for
		    CSC418, SPRING 2005

		implements scene_object.h

***********************************************************/

#include <cmath>
#include <iostream>
#include "scene_object.h"

bool UnitSquare::intersect( Ray3D& ray, const Matrix4x4& worldToModel,
		const Matrix4x4& modelToWorld ) {
	// TODO: implement intersection code for UnitSquare, which is
	// defined on the xy-plane, with vertices (0.5, 0.5, 0), 
	// (-0.5, 0.5, 0), (-0.5, -0.5, 0), (0.5, -0.5, 0), and normal
	// (0, 0, 1).
	//
	// Your goal here is to fill ray.intersection with correct values
	// should an intersection occur.  This includes intersection.point, 
	// intersection.normal, intersection.none, intersection.t_value.   
	//
	// HINT: Remember to first transform the ray into object space  
	// to simplify the intersection test.
    
    // Transform ray to object space
    Point3D origin = worldToModel * ray.origin;
    Vector3D dir   = worldToModel * ray.dir;
    
    // Normal to the plain
    Vector3D n(0,0,1);

    // Find intersection point t with the plane    
    double t;
    double denom;
    Point3D p;

    denom = dir.dot(n);
    
    if(denom==0){
        // the ray and the plane are parallel
        return false;
    }
    
    //t = -(n.dot(origin.toVector3D()));
    //t = t / denom;

    t = -(origin[2]/dir[2]);
    p = origin + (t * dir);
   
    if(t<=0){
        // intersection behind the view plane
        return false;
    }
        
 
    // Check whether point at t is within the boundry
    if(p[0] >= -0.5 && p[0] <= 0.5 &&
       p[1] >= -0.5 && p[1] <= 0.5)
    {
        if(ray.intersection.none || ray.intersection.t_value > t) {
            ray.intersection.t_value = t;
            ray.intersection.point  = modelToWorld * p;
            ray.intersection.normal = transNorm(worldToModel,n);
            ray.intersection.normal.normalize();
            ray.intersection.none   = false;
            return true; 
        }
    }

	return false;
}

bool UnitSphere::intersect( Ray3D& ray, const Matrix4x4& worldToModel,
		const Matrix4x4& modelToWorld ) {
	// TODO: implement intersection code for UnitSphere, which is centred 
	// on the origin.  
	//
	// Your goal here is to fill ray.intersection with correct values
	// should an intersection occur.  This includes intersection.point, 
	// intersection.normal, intersection.none, intersection.t_value.   
	//
	// HINT: Remember to first transform the ray into object space  
	// to simplify the intersection test.

    // Transform ray to object space
    Point3D origin = worldToModel * ray.origin;
    Vector3D dir   = worldToModel * ray.dir;
    Vector3D originv(origin.toVector3D());
    
    double a,b,c;
    double disc, disc_sqrt;
    double q,t0,t1,t;

    
    a  = dir.dot(dir);
    b  = 2 * (dir.dot(originv));
    c  = (originv.dot(originv)) - 1; // r^2=1
    
    disc = (b*b) - (4*a*c); 

    // if discriminant is negative, ray does not intersect
    if(disc<0)
        return false;
    
    disc_sqrt = sqrt(disc);
    
   /*************
    * Avoiding catastrophic cancelation
    ************* 
    if(b<0)
        q = -b + disc_sqrt;
    else
        q = -b - disc_sqrt;

    t0 = q / a;
    t1 = c / q;
    *************/

    // Find the roots
    t0 = (-b + disc_sqrt)/(2*a);
    t1 = (-b - disc_sqrt)/(2*a);

    // Find the first intersection
    if((t0 > 0 ) && (t1 < 0 || t0 <  t1 )){
        // t0 is the closest visible intersection
        t = t0;
    }
    else if((t1 > 0 ) && (t0 < 0 || t1 <  t0 )){
        // t1 is the closest visible intersection
        t = t1;
    } else {
        // No visible intersection
        return false;
    }

    // Update ray.intersection if this is the closest intersection
    if(ray.intersection.none || ray.intersection.t_value > t) {
        
        ray.intersection.t_value = t;
        ray.intersection.none = false;
        
        // intersection point and normal in object space
        ray.intersection.point  = origin + (t * dir);
        ray.intersection.normal = ray.intersection.point.toVector3D();
        
        // Transform intersection point and normal to world space
        ray.intersection.point  = modelToWorld * ray.intersection.point;
        ray.intersection.normal = transNorm(worldToModel,ray.intersection.normal);
        ray.intersection.normal.normalize();
        return true;
    }
    
    return false;

}

