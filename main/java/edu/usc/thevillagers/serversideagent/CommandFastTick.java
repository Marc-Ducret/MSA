package edu.usc.thevillagers.serversideagent;

import net.minecraft.command.CommandBase;
import net.minecraft.command.CommandException;
import net.minecraft.command.ICommandSender;
import net.minecraft.server.MinecraftServer;
import net.minecraft.util.text.TextComponentString;

public class CommandFastTick extends CommandBase {

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
		int ticks = parseInt(args[0]);
		long startTime = System.currentTimeMillis();
		for(int t = 0; t < ticks; t++)
			server.tick();
		long duration = System.currentTimeMillis() - startTime;
		float tickP = duration / (float) ticks;
		float tps = ticks / (duration / 1000F);
		sender.sendMessage(new TextComponentString(String.format("%d ticks completed in %d ms (avg: %.1f ms - %.1f TPS)", ticks, duration, tickP, tps)));
	}
	
	@Override
	public int getRequiredPermissionLevel() {
		return 2;
	}
}