package edu.usc.thevillagers.serversideagent.command;

import net.minecraft.command.CommandBase;
import net.minecraft.command.CommandException;
import net.minecraft.command.ICommandSender;
import net.minecraft.server.MinecraftServer;
import net.minecraft.util.text.TextComponentString;
import net.minecraftforge.common.MinecraftForge;
import net.minecraftforge.fml.common.eventhandler.SubscribeEvent;
import net.minecraftforge.fml.common.gameevent.TickEvent.Phase;
import net.minecraftforge.fml.common.gameevent.TickEvent.ServerTickEvent;

/**
 * Command that displays the current number of 'ticks per second'.
 */
public class CommandTPS extends CommandBase {

	public float avgTickPeriod = 50;
    private long lastTickTime = -1;
	
	public CommandTPS() {
		MinecraftForge.EVENT_BUS.register(this);
	}
	
	@Override
	public String getName() {
		return "tps";
	}

	@Override
	public String getUsage(ICommandSender sender) {
		return "/tps";
	}

	@Override
	public void execute(MinecraftServer server, ICommandSender sender, String[] args) throws CommandException {
		sender.sendMessage(new TextComponentString(String.format("Tick period: %.1f ms; TPS: %.1f", avgTickPeriod, 1000F / avgTickPeriod)));
	}
	
	@Override
	public int getRequiredPermissionLevel() {
		return 2;
	}
	
	@SubscribeEvent
    public void serverTick(ServerTickEvent event) {
    	if(event.phase == Phase.END) {
    		updateTPS();
    	}
    }
	
	private void updateTPS() {
    	long time = System.currentTimeMillis();
		if(lastTickTime >= 0) {
			avgTickPeriod = .9F * avgTickPeriod + .1F * (time - lastTickTime);
		}
		lastTickTime = time;
    }
}
