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

public class CommandFastTick extends CommandBase {
	
	private int requestFastTicks = 0;
    private ICommandSender requestSender = null;
    private boolean fastTicking = false;
	
	public CommandFastTick() {
		MinecraftForge.EVENT_BUS.register(this);
	}

	@Override
	public String getName() {
		return "ft";
	}

	@Override
	public String getUsage(ICommandSender sender) {
		return "/ft <ticks>";
	}

	@Override
	public void execute(MinecraftServer server, ICommandSender sender, String[] args) throws CommandException {
		if(args[0].equalsIgnoreCase("stop")) {
			stopfastTick();
		} else if(args[0].equalsIgnoreCase("start")) {
			requestFastTick(-1, sender);
		} else {
			int ticks = parseInt(args[0]);
			requestFastTick(ticks, sender);
		}
	}
	
	@Override
	public int getRequiredPermissionLevel() {
		return 2;
	}
	
	@SubscribeEvent
    public void serverTick(ServerTickEvent event) {
    	if(event.phase == Phase.END) {
    		if(requestFastTicks != 0) {
    			fastTick(requestFastTicks, requestSender);
    		}
    	}
    }
	
	public void requestFastTick(int ticks, ICommandSender sender) {
    	if(ticks != 0 && sender != null && !fastTicking) {
    		requestFastTicks = ticks;
    		requestSender = sender;
    	}
    }
    
    public void stopfastTick() {
    	fastTicking = false;
    }
    
    private void fastTick(int ticks, ICommandSender sender) {
    	requestFastTicks = 0;
    	fastTicking = true;
    	long startTime = System.currentTimeMillis();
    	int t = 0;
		for(; t != ticks && fastTicking; t++) {
			if(sender.getServer().getTickCounter() % 2 == 0) sender.getServer().tick(); //ensure that we stick to an odd value
			else optimizedTick();
		}
		fastTicking = false;
		long duration = System.currentTimeMillis() - startTime;
		float tickP = duration / (float) t;
		float tps = t / (duration / 1000F);
		sender.sendMessage(new TextComponentString(String.format("%d ticks completed in %d ms (avg: %.1f ms - %.1f TPS)", t, duration, tickP, tps)));
    }
    
    private void optimizedTick() {
    	MinecraftServer serv = net.minecraftforge.fml.common.FMLCommonHandler.instance().getMinecraftServerInstance();
    	serv.profiler.startSection("root");
    	serv.profiler.startSection("tickEvent");
    	net.minecraftforge.fml.common.FMLCommonHandler.instance().onPreServerTick();
    	serv.profiler.endSection();
        serv.updateTimeLightAndEntities();
        serv.profiler.startSection("tickEvent");
        net.minecraftforge.fml.common.FMLCommonHandler.instance().onPostServerTick();
        serv.profiler.endSection();
        serv.profiler.endSection();
    }
}