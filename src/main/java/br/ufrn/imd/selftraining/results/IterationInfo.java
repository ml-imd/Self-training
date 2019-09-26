package br.ufrn.imd.selftraining.results;

public class IterationInfo {
	
	private int addedTolabeled;

	public IterationInfo(int addedTolabeled) {
		this.addedTolabeled = addedTolabeled;
	}
	
	public int getAddedTolabeled() {
		return addedTolabeled;
	}

	public void setAddedTolabeled(int addedTolabeled) {
		this.addedTolabeled = addedTolabeled;
	}
	
}
