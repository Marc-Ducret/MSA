package edu.usc.thevillagers.serversideagent.recording;

import java.util.ArrayList;
import java.util.List;

import net.minecraft.world.World;

public class WorldRecord {
	
	public final World world;
	
	private List<List<RecordEvent>> events;
	private List<RecordEvent> currentTickEvents;
	private int replayT = -1;
	
	public WorldRecord(World world) {
		this.world = world;
		events = new ArrayList<>();
	}
	
	public void startRecordTick() {
		currentTickEvents = new ArrayList<>();
	}
	
	public void recordEvent(RecordEvent event) {
		currentTickEvents.add(event);
	}
	
	public void endRecordTick() {
		if(currentTickEvents == null) return;
		for(RecordEvent e : currentTickEvents) System.out.println(e);
		events.add(currentTickEvents);
	}
	
	public void startReplay() {
		replayT = 0;
	}
	
	public void replayTick() {
		if(replayT >= 0 && replayT < events.size()) {
			List<RecordEvent> tickEvents = events.get(replayT++);
			for(RecordEvent e : tickEvents) {
				e.replay(this);
			}
		}
	}
}
