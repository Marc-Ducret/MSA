package edu.usc.thevillagers.serversideagent;

import edu.usc.thevillagers.serversideagent.recording.RecordEventBlockChange;
import edu.usc.thevillagers.serversideagent.recording.WorldRecord;
import net.minecraft.command.CommandBase;
import net.minecraft.command.CommandException;
import net.minecraft.command.ICommandSender;
import net.minecraft.server.MinecraftServer;
import net.minecraft.util.text.TextComponentString;
import net.minecraft.world.WorldServer;
import net.minecraftforge.common.MinecraftForge;
import net.minecraftforge.event.world.BlockEvent;
import net.minecraftforge.fml.common.eventhandler.SubscribeEvent;
import net.minecraftforge.fml.common.gameevent.TickEvent.ServerTickEvent;

public class CommandRecord extends CommandBase {
	
	private WorldRecord record;
	
	private static enum State{IDLE, RECORDING, REPLAYING}
	private State state = State.IDLE;

	public CommandRecord() {
		MinecraftForge.EVENT_BUS.register(this);
	}
	
	@Override
	public String getName() {
		return "rec";
	}

	@Override
	public String getUsage(ICommandSender sender) {
		return "/rec";
	}

	@Override
	public void execute(MinecraftServer server, ICommandSender sender, String[] args) throws CommandException {
		switch(args[0]) {
		case "start":
			if(state != State.IDLE) throw new CommandException("Already "+state);
			sender.sendMessage(new TextComponentString("Recording started"));
			state = State.RECORDING;
			record = new WorldRecord((WorldServer) sender.getEntityWorld());
			break;
		case "stop":
			if(state != State.RECORDING) throw new CommandException("Actually "+state);
			sender.sendMessage(new TextComponentString("Recording stopped"));
			state = State.IDLE;
			break;
		case "replay":
			if(state != State.IDLE) throw new CommandException("Already "+state);
			sender.sendMessage(new TextComponentString("Replay..."));
			state = State.REPLAYING;
			record.replay();
			sender.sendMessage(new TextComponentString("Replay done"));
			state = State.IDLE;
			break;
		}
	}
	
	@Override
	public int getRequiredPermissionLevel() {
		return 2;
	}
	
	@SubscribeEvent
    public void serverTick(ServerTickEvent event) {
		switch(event.phase) {
		case START:
			if(state == State.RECORDING) record.startRecordTick();
			if(state == State.REPLAYING) record.replayTick();
			break;
		case END:
			if(state == State.RECORDING) record.endRecordTick();
			break;
		}
    }
	
	@SubscribeEvent
	public void blockChange(BlockEvent.NeighborNotifyEvent event) {
		if(state == State.RECORDING) record.recordEvent(new RecordEventBlockChange(event.getPos(), event.getState()));
	}
}
