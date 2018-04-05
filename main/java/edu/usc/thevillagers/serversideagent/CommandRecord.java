package edu.usc.thevillagers.serversideagent;

import edu.usc.thevillagers.serversideagent.recording.RecordEventBlockChange;
import edu.usc.thevillagers.serversideagent.recording.WorldRecord;
import net.minecraft.command.CommandBase;
import net.minecraft.command.CommandException;
import net.minecraft.command.ICommandSender;
import net.minecraft.server.MinecraftServer;
import net.minecraft.util.text.TextComponentString;
import net.minecraftforge.common.MinecraftForge;
import net.minecraftforge.event.world.BlockEvent;
import net.minecraftforge.fml.common.eventhandler.SubscribeEvent;
import net.minecraftforge.fml.common.gameevent.TickEvent.ServerTickEvent;

public class CommandRecord extends CommandBase {
	
	private WorldRecord record;
	private boolean recording;

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
			sender.sendMessage(new TextComponentString("Recording started"));
			record = new WorldRecord(sender.getEntityWorld());
			recording = true;
			break;
		case "stop":
			sender.sendMessage(new TextComponentString("Recording stopped"));
			recording = false;
			break;
		case "replay":
			sender.sendMessage(new TextComponentString("Replay..."));
			record.startReplay();
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
			if(recording) record.startRecordTick();
			if(record != null) record.replayTick();
			break;
		case END:
			if(recording) record.endRecordTick();
			break;
		}
    }
	
	@SubscribeEvent
	public void blockChange(BlockEvent.NeighborNotifyEvent event) {
		if(recording) record.recordEvent(new RecordEventBlockChange(event.getPos(), event.getState()));
	}
}
