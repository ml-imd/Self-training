package br.ufrn.imd.selftraining.bc;

import br.ufrn.imd.selftraining.enums.BaseClassifierType;
import br.ufrn.imd.selftraining.enums.ClassifierType;

public class DecisionTable extends Classifier {

	private String e;
	private String s;
	private boolean i;
	private int x;
	
	public DecisionTable(){
		super();
		this.name = BaseClassifierType.DECISION_TABLE.getInfo();
		this.setClassifierType(ClassifierType.BASE_CLASSIFIER);
	}
	
	@Override
	public void setParametersByDefault() {
		this.e = new String("acc");
		this.s = new String("BestFirst -D 1 -N 5");
		this.i = false;
		this.x = 1;
	}

	public void setParameters(String e, String s, boolean i, int x) {
		this.e = new String(e);
		this.s = new String(s);
		this.i = i;
		this.x = x;
	}
	
	@Override
	public void buildClassifierId() {
		// TODO Auto-generated method stub

	}

	public String getE() {
		return e;
	}

	public void setE(String e) {
		this.e = e;
	}

	public String getS() {
		return s;
	}

	public void setS(String s) {
		this.s = s;
	}

	public boolean isI() {
		return i;
	}

	public void setI(boolean i) {
		this.i = i;
	}

	public int getX() {
		return x;
	}

	public void setX(int x) {
		this.x = x;
	}	
}
