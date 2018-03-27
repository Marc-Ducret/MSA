package edu.usc.thevillagers.serversideagent;

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
    	event.registerServerCommand(new CommandTPS(this));
    }
    
    @SubscribeEvent
    public void serverTick(ServerTickEvent event) {
    	if(event.phase == Phase.END) {
    		long time = System.currentTimeMillis();
    		if(lastTickTime >= 0) {
    			avgTickPeriod = .9F * avgTickPeriod + .1F * (time - lastTickTime);
    		}
    		lastTickTime = time;
    	}
    }
}
