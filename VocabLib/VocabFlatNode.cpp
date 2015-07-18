/* VocabFlatNode.cpp */

#include "VocabTree.h"

#include "../lib/ann_1.1_char/include/ANN/ANN.h"

using namespace ann_1_1_char;

#define NUM_NNS 1 // 3

unsigned long VocabTreeFlatNode::PushAndScoreFeature(
    uint8_t *descriptor, unsigned int index, int bf, int dim, bool add,
    map<unsigned long, float> &node_scores) {
  int nn_idx[NUM_NNS];
  ANNdist distsq[NUM_NNS];

  annMaxPtsVisit(256);

  m_tree->annkPriSearch(descriptor, NUM_NNS, nn_idx, distsq, 0.0);

  unsigned long r = m_children[nn_idx[0]]->PushAndScoreFeature(
      descriptor, index, bf, dim, add, node_scores);

  return r;
}

/* Create a search tree for the given set of keypoints */
void VocabTreeFlatNode::BuildANNTree(int num_leaves, int dim) {
  // unsigned long mem_size = num_leaves * dim;
  // unsigned char *desc = new unsigned char[mem_size];

  /* Create a new array of points */
  ANNpointArray pts = annAllocPts(num_leaves, dim);

  unsigned long id = 0;
  FillDescriptors(num_leaves, dim, id, pts[0]);

  /* Create a search tree for k2 */
  m_tree = new ANNkd_tree(pts, num_leaves, dim, 16);
}
