#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <vector>
#define PATHS 12
#define ENDPOINTS 9910

int main() {
  std::ifstream bigFile("./DRB1-3123_Dist.bin");
  std::cout << "reading data from DRB1-3123_Dist.bin..." << std::endl;
  constexpr size_t bufferSize = PATHS * ENDPOINTS * ENDPOINTS * sizeof(float);
  std::unique_ptr<char[]> buffer(new char[bufferSize]);
  while (bigFile) {
    bigFile.read(buffer.get(), bufferSize);
  }

  float *array;
  array = (float *)malloc(sizeof(int) * PATHS * ENDPOINTS * ENDPOINTS);
  // float avgPathLength = 0.0;
  std::vector<std::pair<int, int>> close_pairs;
  for (int p = 0; p < PATHS; p++) {
    for (int i = 0; i < ENDPOINTS; i++) {
      for (int j = 0; j < ENDPOINTS; j++) {
        float dist = *((float *)&buffer[4 * (p * ENDPOINTS * ENDPOINTS +
                                             i * ENDPOINTS + j)]);
        array[p * ENDPOINTS * ENDPOINTS + i * ENDPOINTS + j] = dist;
        // avgPathLength += dist;
        if (dist > 0.0 && dist < 1e-8) {
          close_pairs.push_back(std::make_pair(i, j));
        }
      }
    }
  }

  // avgPathLength /= (float)PATHS;

  std::vector<std::set<int>> merged_points;
  std::map<int, int> point_to_merged;
  for (auto sorted_pair : close_pairs) {
    int a = std::get<0>(sorted_pair);
    int b = std::get<1>(sorted_pair);
    // go through all the sets and see if a or b is in any of them
    bool inserted = false;
    int inserted_index = 0;
    for (auto &set : merged_points) {
      if (set.find(a) != set.end() || set.find(b) != set.end()) {
        set.insert(a);
        set.insert(b);
        inserted = true;
        break;
      }
      inserted_index++;
    }
    if (inserted) {
      point_to_merged.insert(std::make_pair(a, inserted_index));
      point_to_merged.insert(std::make_pair(b, inserted_index));
    } else {
      std::set<int> new_set;
      new_set.insert(a);
      new_set.insert(b);
      merged_points.push_back(new_set);
      inserted_index = merged_points.size() - 1;
      point_to_merged.insert(std::make_pair(a, inserted_index));
      point_to_merged.insert(std::make_pair(b, inserted_index));
    }
  }

  // fill in the rest of the points
  for (int i = 0; i < ENDPOINTS; i++) {
    if (point_to_merged.find(i) == point_to_merged.end()) {
      std::set<int> new_set;
      new_set.insert(i);
      merged_points.push_back(new_set);
      point_to_merged.insert(std::make_pair(i, merged_points.size() - 1));
    }
  }

  int unique_points = merged_points.size();
  std::cout << "merged_points.size() = " << merged_points.size() << std::endl; // 3521
  std::cout << "point_to_merged.size() = " << point_to_merged.size()
            << std::endl;

  int *weightedEdgeMatrix =
      (int *)malloc(sizeof(int) * unique_points * unique_points);
  // initialize weightedEdgeMatrix
  for (int i = 0; i < unique_points; i++) {
    for (int j = 0; j < unique_points; j++) {
      weightedEdgeMatrix[i * unique_points + j] = 0;
    }
  }
  for (int i = 0; i < ENDPOINTS; i += 2) {
    int start_index = point_to_merged[i];
    int end_index = point_to_merged[i + 1];
    for (int p = 0; p < PATHS; p++) {
      float dist = array[p * ENDPOINTS * ENDPOINTS + i * ENDPOINTS + i + 1];
      if (dist > 0.0) {
        weightedEdgeMatrix[start_index * unique_points + end_index] = (int)dist;
        weightedEdgeMatrix[end_index * unique_points + start_index] = (int)dist;
      }
    }
  }

  // write unique point connection matrix
  std::cout << "writing data to weighted_edge.bin..." << std::endl;
  FILE *file = fopen("./weighted_edge.bin", "wb");
  for (int i = 0; i < unique_points; i++) {
    for (int j = 0; j < unique_points; j++) {
      fwrite(&weightedEdgeMatrix[i * unique_points + j], sizeof(int), 1, file);
    }
  }

  /*
  // initialize distanceMatrix
  int *distanceMatrix = (int *)malloc(sizeof(int) * unique_points * unique_points);
  for (int i = 0; i < unique_points; i++) {
    for (int j = 0; j < unique_points; j++) {
      distanceMatrix[i * unique_points + j] = 0;
    }
  }

  for (int i = 0; i < ENDPOINTS; i++) {
    for (int j = i+1; j < ENDPOINTS; j++) {
      int start_index = point_to_merged[i];
      int end_index = point_to_merged[j];
      if (start_index == end_index) {
        continue;
      }

      // find the shortest distance
      int shortest_dist = -1;
      for (int p = 0; p < PATHS; p++) {
        float dist = array[p * ENDPOINTS * ENDPOINTS + i * ENDPOINTS + j];
        if (dist > 1e-3) {
          if (dist < shortest_dist) {
            shortest_dist = (int)dist;
          }
        }
      }
      shortest_dist = shortest_dist > 0 ? shortest_dist : 10000;
      distanceMatrix[start_index * unique_points + end_index] = shortest_dist;
      distanceMatrix[end_index * unique_points + start_index] = shortest_dist;
    }
  }

  // write distance matrix
  std::cout << "writing data to distance.bin..." << std::endl;
  FILE *dfile = fopen("./distance.bin", "wb");
  for (int i = 0; i < unique_points; i++) {
    for (int j = 0; j < unique_points; j++) {
      if (i == 0) {
        std::cout << distanceMatrix[i * unique_points + j] << " ";
      }
      fwrite(&distanceMatrix[i * unique_points + j], sizeof(int), 1, dfile);
    }
    if (i == 0) {
      std::cout << std::endl;
    }
  }

  */
  return 0;
}