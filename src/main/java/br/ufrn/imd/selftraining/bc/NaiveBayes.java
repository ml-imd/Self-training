package br.ufrn.imd.selftraining.bc;

import br.ufrn.imd.selftraining.enums.BaseClassifierType;
import br.ufrn.imd.selftraining.enums.ClassifierType;
import br.ufrn.imd.selftraining.utils.Encryptor;

public class NaiveBayes extends Classifier {

	private boolean d;
	private boolean k;
	private boolean o;

	public NaiveBayes() {
		super();
		this.name = BaseClassifierType.NAIVE_BAYES.getInfo();
		this.classifierType = ClassifierType.BASE_CLASSIFIER;
	}

	/**
	 * @param D boolean - To use supervised discretization to process numeric
	 *          attributes. (default false);
	 * @param K boolean - To use kernel density estimator rather than normal
	 *          distribution for numeric attributes. (default false);
	 * @param O boolean - To display model in old format. (default false);
	 */
	public void setParameters(boolean D, boolean K) {
		this.d = D;
		this.k = K;
	}

	@Override
	public void setParametersByDefault() {
		this.d = false;
		this.k = false;
		this.o = false;
	}

	@Override
	public void buildClassifierId() {
		String id = "-D" + getD() + "-K" + getK() + "-O" + getO();
		this.classifierId = new String(Encryptor.encryptSh1(id));
	}

	public boolean getD() {
		return d;
	}

	public void setD(boolean d) {
		this.d = d;
	}

	public boolean getK() {
		return k;
	}

	public void setK(boolean k) {
		this.k = k;
	}

	public boolean getO() {
		return o;
	}

	public void setO(boolean o) {
		this.o = o;
	}

}
