package edu.usc.thevillagers.serversideagent;

import net.minecraft.command.CommandBase;
import net.minecraft.command.CommandException;
import net.minecraft.command.ICommandSender;
import net.minecraft.server.MinecraftServer;
import net.minecraft.util.text.TextComponentString;

public class CommandTPS extends CommandBase {

	private final ServerSideAgentMod mod;
	
	public CommandTPS(ServerSideAgentMod mod) {
		this.mod = mod;
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
		sender.sendMessage(new TextComponentString(String.format("Tick period: %.1f ms; TPS: %.1f", mod.avgTickPeriod, 1000F / mod.avgTickPeriod)));
	}
	
	@Override
	public int getRequiredPermissionLevel() {
		return 2;
	}
}
