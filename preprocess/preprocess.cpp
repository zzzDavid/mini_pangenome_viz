#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <vector>
#define PATHS 12
#define ENDPOINTS 9910

int main() {
  std::ifstream bigFile("./data.bin");
  std::cout << "reading data from data.bin..." << std::endl;
  constexpr size_t bufferSize = PATHS * ENDPOINTS * ENDPOINTS * sizeof(float);
  std::unique_ptr<char[]> buffer(new char[bufferSize]);
  while (bigFile) {
    bigFile.read(buffer.get(), bufferSize);
  }

  float *array;
  array = (float *)malloc(sizeof(int) * PATHS * ENDPOINTS * ENDPOINTS);

  std::vector<std::pair<int, int>> close_pairs;
  for (int p = 0; p < PATHS; p++) {
    for (int i = 0; i < ENDPOINTS; i++) {
      for (int j = 0; j < ENDPOINTS; j++) {
        float dist = *((float *)&buffer[4 * (p * ENDPOINTS * ENDPOINTS +
                                             i * ENDPOINTS + j)]);
        array[p * ENDPOINTS * ENDPOINTS + i * ENDPOINTS + j] = dist;
        if (dist > 0.0 && dist < 1e-8) {
          close_pairs.push_back(std::make_pair(i, j));
        }
      }
    }
  }

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

  int *distanceMatrix =
      (int *)malloc(sizeof(int) * unique_points * unique_points);
  // initialize distanceMatrix
  for (int i = 0; i < unique_points; i++) {
    for (int j = 0; j < unique_points; j++) {
      distanceMatrix[i * unique_points + j] = 0;
    }
  }
  for (int i = 0; i < ENDPOINTS; i += 2) {
    int start_index = point_to_merged[i];
    int end_index = point_to_merged[i + 1];
    for (int p = 0; p < PATHS; p++) {
      float dist = array[p * ENDPOINTS * ENDPOINTS + i * ENDPOINTS + i + 1];
      if (dist > 0.0) {
        distanceMatrix[start_index * unique_points + end_index] = (int)dist;
        distanceMatrix[end_index * unique_points + start_index] = (int)dist;
      }
    }
  }

  // write unique point connection matrix
  std::cout << "writing data to distance.bin..." << std::endl;
  FILE *file = fopen("./distance.bin", "wb");
  for (int i = 0; i < unique_points; i++) {
    for (int j = 0; j < unique_points; j++) {
      if (i == 0)
        std::cout << distanceMatrix[i * unique_points + j] << " ";
      fwrite(&distanceMatrix[i * unique_points + j], sizeof(int), 1, file);
    }
    if (i == 0)
      std::cout << std::endl;
  }
  return 0;
}