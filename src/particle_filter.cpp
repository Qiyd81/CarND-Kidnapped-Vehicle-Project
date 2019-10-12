/**
 * particle_filter.cpp
 *
 * Created on: Oct. 11, 2019
 * Author: Yundong Qi
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

static std::default_random_engine gen;   //create random engine to be used in below functions
void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 100; //set the particles quantity

  // define normal distributions for initial position
  std::normal_distribution<double> X_init_dist(x, std[0]);
  std::normal_distribution<double> Y_init_dist(y, std[1]);
  std::normal_distribution<double> Theta_init_dist(theta, std[2]);

  // initialize particles
  for (int i = 0; i < num_particles; i++) {
    Particle p;
    p.id = i;
    p.x = X_init_dist(gen);
    p.y = Y_init_dist(gen);
    p.theta = Theta_init_dist(gen);
    p.weight = 1.0;

    particles.push_back(p);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  //define normal distribution noise for predicted position
  std::normal_distribution<double> P_noise_x(0, std_pos[0]);
  std::normal_distribution<double> P_noise_y(0, std_pos[1]);
  std::normal_distribution<double> P_noise_theta(0, std_pos[2]);
  
  //predict each particles position
  for (int i = 0; i < num_particles; i++) {

    // predict next step position
    // if yaw_rate < 0.000001, assume no yaw change; otherwise, use the yaw_rate to predict new position
    if (fabs(yaw_rate) < 0.000001) {
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    }
    else {
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }

    // add position noise
    particles[i].x += P_noise_x(gen);
    particles[i].y += P_noise_y(gen);
    particles[i].theta += P_noise_theta(gen);
  }
}
  

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  
  //loop over all observations to find each observation's closest map landmark id. These observations are already transformed. And the predicted stores all the potential landmarks that can be detected from the particle's current position.
  for (unsigned int i = 0; i < observations.size(); i++) {
    
    // take each observation
    LandmarkObs obs_lm = observations[i];

    // define minimum distance placeholder variable and maximize the value to avoid missing any comparasion
    double min_dist = std::numeric_limits<double>::max();

    // define one placeholder variable to store the landmark id and init with -1
    int map_id = -1;
    
    //loop over all predicted landmarks to find the closest one
    for (unsigned int j = 0; j < predicted.size(); j++) {
      // take each predicted landmark
      LandmarkObs pred_lm = predicted[j];
      
      // calculate the distance between observated and predicted landmarks
      double op_dist = dist(obs_lm.x, obs_lm.y, pred_lm.x, pred_lm.y);

      // compare the op_dist with the min_dist, to find the less one
      if (op_dist < min_dist) {
        min_dist = op_dist;            //set the min_dist to the less one
        map_id = pred_lm.id;           //change the id to the predicted landmark with less distance
      }
    }
    // set the observation's id to the predicted landmark's id with shortest distance
    observations[i].id = map_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  //loop over all particles
  for (int i = 0; i < num_particles; i++) {

    // get each particle's parameters
    double part_x = particles[i].x;
    double part_y = particles[i].y;
    double part_theta = particles[i].theta;

    // transform the observations to map coordinates system
    //define a new vector to store the transformed observations
    vector<LandmarkObs> trans_obs;
    //loop over all observations to transform
    for (unsigned int j = 0; j < observations.size(); j++) {
      double trans_x = part_x + cos(part_theta)*observations[j].x - sin(part_theta)*observations[j].y;
      double trans_y = part_y + sin(part_theta)*observations[j].x + cos(part_theta)*observations[j].y;
      trans_obs.push_back(LandmarkObs{observations[j].id, trans_x, trans_y});
    }

    // define a vector to store all the map landmarks that can be detected from the particle, within sensor range of the particle
    vector<LandmarkObs> predicted_lms;

    // loop over all map landmarks to find the ones can be detected
    for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      // take the id, x, y of each landmark
      float lm_x = map_landmarks.landmark_list[j].x_f;
      float lm_y = map_landmarks.landmark_list[j].y_f;
      int lm_id = map_landmarks.landmark_list[j].id_i;
      
      // find the landmarks inside the sensor_range, assume a rectangular sensor area
      if (fabs(lm_x - part_x) <= sensor_range && fabs(lm_y - part_y) <= sensor_range) {

        // attach the landmark to the vector
        predicted_lms.push_back(LandmarkObs{lm_id, lm_x, lm_y});
      }
    }
    
    // perform dataAssociation for the predicted landmarks and transformed observations on current particle
    dataAssociation(predicted_lms, trans_obs);

    // reinit particle weight to 1.0, this is important, otherwise the weight will accumulate with each time step
    particles[i].weight = 1.0;

    // loop over all transformed observations to update the weight
    for (unsigned int j = 0; j < trans_obs.size(); j++) {
      // take the associated landmark id associated with each observation
      int ass_id = trans_obs[j].id;
      
      // placeholders for transformed observation and associated predicted landmark coordinates
      double o_x, o_y, pr_x, pr_y;
      // take the observation x, y
      o_x = trans_obs[j].x;
      o_y = trans_obs[j].y;
      // take the associated map landmark's x, y, directly from the landmark list, as the landmark id-1 is the landmark index.
      pr_x = map_landmarks.landmark_list[ass_id-1].x_f;
      pr_y = map_landmarks.landmark_list[ass_id-1].y_f;

      //      // another method to get the x,y coordinates of the predicted_lms associated with the current observation, more robust, but might be slower
      //      for (unsigned int k = 0; k < predicted_lms.size(); k++) {
      //        if (predicted_lms[k].id == trans_obs[j].id) {
      //          pr_x = predicted_lms[k].x;
      //          pr_y = predicted_lms[k].y;
      //        }
      //      }

      // calculate this observation's weight with multivariate Gaussian
      double s_x = std_landmark[0];
      double s_y = std_landmark[1];
      double obs_w = (1/(2*M_PI*s_x*s_y)) * exp(-(pow(o_x-pr_x,2)/(2*pow(s_x,2))+(pow(o_y-pr_y,2)/(2*pow(s_y, 2)))));

      // calculate the total observations weight with the product of each observation's weight
      particles[i].weight *= obs_w;
    }
  }
}
  

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  //define the placeholer vector for updated particles
  vector<Particle> up_particles;

  //define one placeholder vector to get all of the current particles' weights
  vector<double> weights;
  for (int i = 0; i < num_particles; i++) {
    weights.push_back(particles[i].weight);
  }

  // generate uniform random starting index to start resampling
  std::uniform_int_distribution<int> int_dist(0, num_particles-1);
  auto index = int_dist(gen);

  // take the max weight
  double max_weight = *max_element(weights.begin(), weights.end());

  // generate uniform random distribution betweeen [0.0, 2*max_weight)
  std::uniform_real_distribution<double> real_dist(0.0, 2*max_weight);

  //init beta
  double beta = 0.0;

  // resample propotional to weights
  for (int i = 0; i < num_particles; i++) {
    beta += real_dist(gen);
    while (beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    up_particles.push_back(particles[index]);
  }
  //update particles
  particles = up_particles;
}
void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
