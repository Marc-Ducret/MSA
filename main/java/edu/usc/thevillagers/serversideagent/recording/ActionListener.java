package edu.usc.thevillagers.serversideagent.recording;

import edu.usc.thevillagers.serversideagent.HighLevelAction;

/**
 * Listener for when a action is made during a replay
 */
public interface ActionListener {

	public void onAction(HighLevelAction action);
}
