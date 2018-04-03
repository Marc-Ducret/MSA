package edu.usc.thevillagers.serversideagent;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import edu.usc.thevillagers.serversideagent.env.Environment;
import net.minecraft.command.ICommandSender;
import net.minecraft.util.text.TextComponentString;
import net.minecraftforge.common.MinecraftForge;
import net.minecraftforge.fml.common.Mod;
import net.minecraftforge.fml.common.Mod.EventHandler;
import net.minecraftforge.fml.common.event.FMLInitializationEvent;
import net.minecraftforge.fml.common.event.FMLPreInitializationEvent;
import net.minecraftforge.fml.common.event.FMLServerStartingEvent;
import net.minecraftforge.fml.common.eventhandler.SubscribeEvent;
import net.minecraftforge.fml.common.gameevent.TickEvent.Phase;
import net.minecraftforge.fml.common.gameevent.TickEvent.ServerTickEvent;

@Mod(modid = ServerSideAgentMod.MODID, name = ServerSideAgentMod.NAME, version = ServerSideAgentMod.VERSION)
public class ServerSideAgentMod {
    public static final String MODID = "serversideagent";
    public static final String NAME = "Server Side Agent";
    public static final String VERSION = "1.0";

    public float avgTickPeriod = 50;
    private long lastTickTime = -1;
    
    private int requestFastTicks = 0;
    private ICommandSender requestSender = null;
    private boolean fastTicking = false;
    
    private List<Environment> envs = new ArrayList<>();
    
    @EventHandler
    public void preInit(FMLPreInitializationEvent event) {
    }

    @EventHandler
    public void init(FMLInitializationEvent event) {
    	MinecraftForge.EVENT_BUS.register(this);
    }
    
    @EventHandler
    public void serverLoad(FMLServerStartingEvent event) {
    	event.registerServerCommand(new CommandSpawn());
    	event.registerServerCommand(new CommandEnvironment(this));
    	event.registerServerCommand(new CommandFastTick(this));
    	event.registerServerCommand(new CommandTPS(this));
    }
    
    private void tickEnvs(Phase phase) {
    	Iterator<Environment> iter = envs.iterator();
		while(iter.hasNext()) {
			Environment env = iter.next();
			try {
				switch(phase) {
				case START:
					env.preTick();
					break;
				case END:
					env.postTick();
					break;
				default:
					break;
				}
			} catch(Exception e) {
				env.terminate();
				iter.remove();
				System.err.println("Env "+env.name+" terminated ("+e+")");
			}
		}
    }
    
    @SubscribeEvent
    public void serverTick(ServerTickEvent event) {
    	tickEnvs(event.phase);
    	if(event.phase == Phase.END) {
    		updateTPS();
    		if(requestFastTicks != 0) {
    			fastTick(requestFastTicks, requestSender);
    		}
    	}
    }
    
    private void updateTPS() {
    	long time = System.currentTimeMillis();
		if(lastTickTime >= 0) {
			avgTickPeriod = .9F * avgTickPeriod + .1F * (time - lastTickTime);
		}
		lastTickTime = time;
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
			sender.getServer().tick();
		}
		long duration = System.currentTimeMillis() - startTime;
		float tickP = duration / (float) t;
		float tps = t / (duration / 1000F);
		sender.sendMessage(new TextComponentString(String.format("%d ticks completed in %d ms (avg: %.1f ms - %.1f TPS)", t, duration, tickP, tps)));
    }
    
    public void addEnv(Environment env) {
    	envs.add(env);
    }
}
