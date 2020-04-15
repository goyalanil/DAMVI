"""
       Diversity-Aware weighted Majority Vote for Imbalanced datasets (DAMVI)

  File:     damvi.py
  Authors:  Anil Goyal (anil.goyal@neclab.eu)


NEC Laboratories Europe GmbH, Copyright (c) <year>, All rights reserved.

       THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.

       PROPRIETARY INFORMATION ---

SOFTWARE LICENSE AGREEMENT

ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY

BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS
LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR
DOWNLOAD THE SOFTWARE.

This is a license agreement ("Agreement") between your academic institution
or non-profit organization or self (called "Licensee" or "You" in this
Agreement) and NEC Laboratories Europe GmbH (called "Licensor" in this
Agreement).  All rights not specifically granted to you in this Agreement
are reserved for Licensor.

RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: Licensor retains exclusive
ownership of any copy of the Software (as defined below) licensed under this
Agreement and hereby grants to Licensee a personal, non-exclusive,
non-transferable license to use the Software for noncommercial research
purposes, without the right to sublicense, pursuant to the terms and
conditions of this Agreement. NO EXPRESS OR IMPLIED LICENSES TO ANY OF
LICENSOR'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. As used in this
Agreement, the term "Software" means (i) the actual copy of all or any
portion of code for program routines made accessible to Licensee by Licensor
pursuant to this Agreement, inclusive of backups, updates, and/or merged
copies permitted hereunder or subsequently supplied by Licensor,  including
all or any file structures, programming instructions, user interfaces and
screen formats and sequences as well as any and all documentation and
instructions related to it, and (ii) all or any derivatives and/or
modifications created or made by You to any of the items specified in (i).

CONFIDENTIALITY/PUBLICATIONS: Licensee acknowledges that the Software is
proprietary to Licensor, and as such, Licensee agrees to receive all such
materials and to use the Software only in accordance with the terms of this
Agreement.  Licensee agrees to use reasonable effort to protect the Software
from unauthorized use, reproduction, distribution, or publication. All
publication materials mentioning features or use of this software must
explicitly include an acknowledgement the software was developed by NEC
Laboratories Europe GmbH.

COPYRIGHT: The Software is owned by Licensor.

PERMITTED USES:  The Software may be used for your own noncommercial
internal research purposes. You understand and agree that Licensor is not
obligated to implement any suggestions and/or feedback you might provide
regarding the Software, but to the extent Licensor does so, you are not
entitled to any compensation related thereto.

DERIVATIVES: You may create derivatives of or make modifications to the
Software, however, You agree that all and any such derivatives and
modifications will be owned by Licensor and become a part of the Software
licensed to You under this Agreement.  You may only use such derivatives and
modifications for your own noncommercial internal research purposes, and you
may not otherwise use, distribute or copy such derivatives and modifications
in violation of this Agreement.

BACKUPS:  If Licensee is an organization, it may make that number of copies
of the Software necessary for internal noncommercial use at a single site
within its organization provided that all information appearing in or on the
original labels, including the copyright and trademark notices are copied
onto the labels of the copies.

USES NOT PERMITTED:  You may not distribute, copy or use the Software except
as explicitly permitted herein. Licensee has not been granted any trademark
license as part of this Agreement.  Neither the name of NEC Laboratories
Europe GmbH nor the names of its contributors may be used to endorse or
promote products derived from this Software without specific prior written
permission.

You may not sell, rent, lease, sublicense, lend, time-share or transfer, in
whole or in part, or provide third parties access to prior or present
versions (or any parts thereof) of the Software.

ASSIGNMENT: You may not assign this Agreement or your rights hereunder
without the prior written consent of Licensor. Any attempted assignment
without such consent shall be null and void.

TERM: The term of the license granted by this Agreement is from Licensee's
acceptance of this Agreement by downloading the Software or by using the
Software until terminated as provided below.

The Agreement automatically terminates without notice if you fail to comply
with any provision of this Agreement.  Licensee may terminate this Agreement
by ceasing using the Software.  Upon any termination of this Agreement,
Licensee will delete any and all copies of the Software. You agree that all
provisions which operate to protect the proprietary rights of Licensor shall
remain in force should breach occur and that the obligation of
confidentiality described in this Agreement is binding in perpetuity and, as
such, survives the term of the Agreement.

FEE: Provided Licensee abides completely by the terms and conditions of this
Agreement, there is no fee due to Licensor for Licensee's use of the
Software in accordance with this Agreement.

DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY
OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR
FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON- INFRINGEMENT.  LICENSEE
BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND
RELATED MATERIALS.

SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is
provided as part of this Agreement.

EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent
permitted under applicable law, Licensor shall not be liable for direct,
indirect, special, incidental, or consequential damages or lost profits
related to Licensee's use of and/or inability to use the Software, even if
Licensor is advised of the possibility of such damage.

EXPORT REGULATION: Licensee agrees to comply with any and all applicable
export control laws, regulations, and/or other laws related to embargoes and
sanction programs administered by law.

SEVERABILITY: If any provision(s) of this Agreement shall be held to be
invalid, illegal, or unenforceable by a court or other tribunal of competent
jurisdiction, the validity, legality and enforceability of the remaining
provisions shall not in any way be affected or impaired thereby.

NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right
or remedy under this Agreement shall be construed as a waiver of any future
or other exercise of such right or remedy by Licensor.

GOVERNING LAW: This Agreement shall be construed and enforced in accordance
with the laws of Germany without reference to conflict of laws principles.
You consent to the personal jurisdiction of the courts of this country and
waive their rights to venue outside of Germany.

ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and
entire agreement between Licensee and Licensor as to the matter set forth
herein and supersedes any previous agreements, understandings, and
arrangements between the parties relating hereto.

       THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.


"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score,precision_score, roc_auc_score, average_precision_score,confusion_matrix, recall_score

from .C_bound_opt import Cbound_opt

class damvi(object):
    """
    This class solves the weighted least square optimization problem.
    It finally returns the weights over classifiers
    """
    def __init__(self, no_models = 100):

        self.no_models = no_models


    def _bagging(self, X_train, y_train, no_models):
        """

        :return:
        """

        learned_models = []
        sss_bagging = StratifiedShuffleSplit(n_splits=no_models, test_size=0.8, random_state=0)
        i = 0
        for train_index_bagging, test_index_bagging in sss_bagging.split(X_train, y_train):
            X_train_sampled = X_train[train_index_bagging]
            y_train_sampled = y_train[train_index_bagging]

            clf = DecisionTreeClassifier()

            clf.fit(X_train_sampled, y_train_sampled)
            learned_models.append(clf)
            predicted_labels_train = clf.predict(X_train)

            if i == 0:
                train_data_predictions = np.array(predicted_labels_train)
                i += 1
            else:
                train_data_predictions = np.vstack((train_data_predictions, predicted_labels_train))
                i += 1

        return train_data_predictions, learned_models


    def _example_reweighing(self, train_data_predictions, y_train, sample_distribution):
        """

        :return:
        """

        mv = np.zeros(train_data_predictions.shape[1])
        weights = np.ones(self.no_models) / self.no_models

        for i in range(0, (self.no_models)):
            mv = mv + weights[i] * train_data_predictions[i]

        mv = np.sign(mv)

        class_considered = 1
        sample_distribution[y_train==class_considered] = sample_distribution[y_train==class_considered]\
                                                         * np.exp(-y_train[y_train==class_considered]
                                                                  * mv[y_train==class_considered])
        sample_distribution = sample_distribution/sum(sample_distribution)

        return sample_distribution


    def fit(self, X_train, y_train):
        """
        Learns weights
        :param self:
        :return:
        """

        # Bagging
        train_data_predictions, self.learned_models = self._bagging(X_train, y_train,  self.no_models)

        #Give more weights to positive examples which are misclassified
        sample_distribution = np.ones(X_train.shape[0]) / X_train.shape[0]
        sample_distribution = self._example_reweighing(train_data_predictions, y_train, sample_distribution)

        # Optimization (Learn weights over classifiers)
        self.weights = Cbound_opt(train_data_predictions.transpose(), y_train, sample_distribution).learn_weights()



    def predict(self, X_test):

        # Predictions (Majority Voting) on test data
        mv = np.zeros(X_test.shape[0])
        prob_test_data_positive = np.zeros(X_test.shape[0])
        prob_test_data_negative = np.zeros(X_test.shape[0])
        for i in range(0, (self.no_models)):
            clf = self.learned_models[i]
            test_data_predictions = clf.predict(X_test)
            predicted_prob_test_array = clf.predict_proba(X_test)
            test_data_probabilities_negative = predicted_prob_test_array[:, 0]
            test_data_probabilities_positive = predicted_prob_test_array[:, 1]

            mv = mv + self.weights[i] * test_data_predictions
            prob_test_data_positive = prob_test_data_positive + self.weights[i] * test_data_probabilities_positive
            prob_test_data_negative = prob_test_data_negative + self.weights[i] * test_data_probabilities_negative

        y_predicted = np.sign(mv)

        return y_predicted

    def predict_proba(self, X_test):

        # Predictions (Majority Voting) on test data
        mv = np.zeros(X_test.shape[0])
        prob_test_data_positive = np.zeros(X_test.shape[0])
        prob_test_data_negative = np.zeros(X_test.shape[0])
        for i in range(0, (self.no_models)):
            clf = self.learned_models[i]
            test_data_predictions = clf.predict(X_test)
            predicted_prob_test_array = clf.predict_proba(X_test)
            test_data_probabilities_negative = predicted_prob_test_array[:, 0]
            test_data_probabilities_positive = predicted_prob_test_array[:, 1]

            mv = mv + self.weights[i] * test_data_predictions
            prob_test_data_positive = prob_test_data_positive + self.weights[i] * test_data_probabilities_positive
            prob_test_data_negative = prob_test_data_negative + self.weights[i] * test_data_probabilities_negative

        y_score = mv

        return y_score

