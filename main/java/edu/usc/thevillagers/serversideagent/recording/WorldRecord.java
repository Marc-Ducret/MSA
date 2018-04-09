package edu.usc.thevillagers.serversideagent.recording;

import java.util.ArrayList;
import java.util.List;

import net.minecraft.server.MinecraftServer;
import net.minecraft.world.WorldServer;
import net.minecraftforge.fml.common.FMLCommonHandler;

public class WorldRecord {
	
	public final WorldServer world;
	
	private List<List<RecordEvent>> events;
	private List<RecordEvent> currentTickEvents;
	private int replayT = -1;
	
	public WorldRecord(WorldServer world) {
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
	
	public void replay() {
		replayT = 0;
		while(replayT < events.size()) {
			replayTick();
			world.tick();
			world.getEntityTracker().tick();
			MinecraftServer serv = FMLCommonHandler.instance().getMinecraftServerInstance();
			serv.getNetworkSystem().networkTick();
			serv.getPlayerList().onTick();
			
			try {
				Thread.sleep(10);
			} catch (InterruptedException e) {}
		}
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
